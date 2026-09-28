#!/usr/bin/env python3
"""Expand the RunningBench MCQ pool: more distractors per question, more questions per segment.

Both modes are text-only — they read the existing `dense_annotations`; no video is
decoded or uploaded. Output goes to a separate work directory and the source
annotations are never modified.

The Floodgate quota measured on this project is a hard 0.5 requests/second
(30/min) regardless of concurrency, so the design minimises CALL COUNT rather
than adding workers: every stage batches all of a segment's questions into one
request, and a shared token bucket paces sending just under the quota so time is
not wasted in 429 backoff.

  --expand-options   pad every choice question up to the target option count,
                     keeping only distractors a second pass judges to be
                     contradicted by the evidence.
  --more-questions   generate additional questions per segment, deliberately
                     different in focus from the ones already present.

Usage:
  python3 expand_mcq.py --expand-options --more-questions --dry-run
  python3 expand_mcq.py --expand-options --more-questions --pilot P01_SlowWalk_Day_Traj01
  python3 expand_mcq.py --expand-options --more-questions
"""
import argparse, hashlib, importlib.util, json, os, random, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEG_DIRS = [
    Path("/mnt/data/data_anno/conductor_release/annotations_60s"),
    Path("/mnt/data/data_anno/runningbench_gap_repair/annotations_60s"),
]
DEFAULT_WORK = Path("/mnt/data/data_anno/runningbench_qa_expansion")

TARGET_SINGLE = 6                # A-F, random baseline 16.7%
TARGET_MULTI = 8                 # A-H
NEW_QUESTIONS_PER_SEGMENT = 3    # on top of the 3 already there
QUOTA_PER_SECOND = 0.48          # measured ceiling is 0.50; stay just under it

LEAK_PHRASES = (
    "explicitly described", "as described", "described in the video", "described throughout",
    "according to the description", "the description", "mentioned in", "stated in",
    "the evidence", "the annotation", "based on the text", "the provided",
)

DISTRACTOR_PROMPT = """You are extending video-understanding benchmark questions with extra wrong options.

EVIDENCE (the only ground truth; one segment of a first-person walking/running video):
{evidence}

For each question below, write exactly the requested number of NEW options. Every new option must be:
- plausible for this kind of video, similar in length and style to the existing options;
- DEFINITIVELY CONTRADICTED by the evidence above — never merely unmentioned, never partly true;
- clearly distinct from every existing option and from your other new options.

QUESTIONS:
{questions}

Return only a JSON object keyed by question id:
{{"new_options": {{"q1": ["...", "..."], "q2": ["..."]}}}}"""

VERIFY_PROMPT = """Judge each numbered statement against the evidence from a first-person video segment.

EVIDENCE:
{evidence}

STATEMENTS:
{statements}

For each statement answer strictly:
- "contradicted" if the evidence positively rules it out;
- "supported" if the evidence indicates it is true;
- "unclear" if the evidence neither confirms nor rules it out.

Return only a JSON object with one verdict per statement, in the same order:
{{"verdicts": ["contradicted", "unclear", ...]}}"""

QUESTION_PROMPT = """Create exactly {count} NEW English benchmark questions from the evidence below.

EVIDENCE (the only ground truth; one segment of a first-person walking/running video):
{evidence}

QUESTIONS ALREADY ASKED about this segment — yours must test something different:
{existing}

Return only a JSON object with the key `qa_pairs`. Every question must contain
question, options, answer, is_multiple_choice and type.
- `type` is one of trajectory-qa, visual-grounded-qa, spatial-temporal-reasoning.
- `options` is a JSON OBJECT keyed by consecutive uppercase letters starting at A,
  e.g. {{"A": "...", "B": "..."}} — not a list.
- Exactly {n_single} questions are single choice with {target_single} options and exactly one answer.
- The rest are multiple choice with {target_multi} options and at least two answers.
- `answer` is a non-empty list of option letters.

Use only the supplied evidence, but write every question as if asking someone who is
WATCHING THE VIDEO and has never seen this text. Never refer to the evidence itself:
no "as described", "explicitly described", "described in the video", "mentioned",
"according to the description", "the evidence". Ask about what happens on screen.

Distractors must be plausible but unambiguously false."""


class RateLimiter:
    """Shared token bucket: the server quota, not worker count, is the bottleneck."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_slot = time.monotonic()

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_slot - now)
            self.next_slot = max(now, self.next_slot) + self.interval
        if wait:
            time.sleep(wait)


class PacedSession:
    """Wrap a requests.Session so the quota paces every outbound attempt.

    Pacing only the logical call is not enough: generate() and
    generate_valid_json() retry internally, and those retries would otherwise
    bypass the limiter, hit 429 and burn 5-120s of backoff each.
    """

    def __init__(self, session, limiter):
        self._session = session
        self._limiter = limiter

    def post(self, *args, **kwargs):
        self._limiter.acquire()
        return self._session.post(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "repair_runningbench_annotations", HERE / "repair_runningbench_annotations.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shortfall_of(item) -> int:
    options, answer = item.get("options"), item.get("answer") or []
    if not isinstance(options, dict) or not options:
        return 0
    if sorted(options) != [chr(ord("A") + i) for i in range(len(options))]:
        return 0                      # corrupted keys: leave alone
    want = TARGET_MULTI if len(answer) > 1 else TARGET_SINGLE
    return max(0, want - len(options))


def make_distractor_validator(wanted: dict):
    def validate(data):
        block = data.get("new_options")
        if not isinstance(block, dict):
            raise ValueError("new_options must be an object keyed by question id")
        for qid, count in wanted.items():
            options = block.get(qid)
            if not isinstance(options, list) or len(options) != count:
                raise ValueError(f"{qid} needs exactly {count} new options")
            if any(not isinstance(o, str) or not o.strip() for o in options):
                raise ValueError(f"{qid} has an empty option")
            if len({o.strip().lower() for o in options}) != count:
                raise ValueError(f"{qid} has duplicate new options")
    return validate


def make_verdict_validator(count):
    def validate(data):
        verdicts = data.get("verdicts")
        if not isinstance(verdicts, list) or len(verdicts) != count:
            raise ValueError(f"verdicts must be a list of exactly {count} entries")
        if any(v not in ("contradicted", "supported", "unclear") for v in verdicts):
            raise ValueError("each verdict must be contradicted, supported or unclear")
    return validate


def make_question_validator(count):
    def validate(data):
        qa = data.get("qa_pairs")
        if not isinstance(qa, list) or len(qa) != count:
            raise ValueError(f"qa_pairs must contain exactly {count} questions")
        for item in qa:
            # The model reliably answers with a positional list of option strings
            # rather than a letter-keyed object. Normalise instead of retrying:
            # position already carries the letter, so the mapping is lossless.
            if isinstance(item.get("options"), list) and all(
                    isinstance(o, str) for o in item["options"]):
                item["options"] = {chr(ord("A") + i): o for i, o in enumerate(item["options"])}
            options, answer = item.get("options"), item.get("answer")
            if not item.get("question"):
                raise ValueError("question text is required")
            if item.get("type") not in ("trajectory-qa", "visual-grounded-qa", "spatial-temporal-reasoning"):
                raise ValueError("invalid type")
            if not isinstance(options, dict):
                raise ValueError("options must be an object")
            if sorted(options) != [chr(ord("A") + i) for i in range(len(options))]:
                raise ValueError("option keys must be consecutive from A")
            if not isinstance(answer, list) or not answer or not set(answer) <= set(options):
                raise ValueError("invalid answer")
            item["is_multiple_choice"] = len(answer) > 1
            want = TARGET_MULTI if len(answer) > 1 else TARGET_SINGLE
            if len(options) != want:
                raise ValueError(f"expected {want} options for this answer arity, got {len(options)}")
            lowered = item["question"].lower()
            leaked = [p for p in LEAK_PHRASES if p in lowered]
            if leaked:
                # Citing the description makes the question unanswerable from the video.
                raise ValueError(f"question refers to the evidence text ({leaked[0]!r})")
        if not any(len(i["answer"]) == 1 for i in qa):
            raise ValueError("need at least one single-choice question")
    return validate


def reletter(item, texts, correct_texts, salt):
    order = list(range(len(texts)))
    seed = hashlib.sha256((item["question"] + salt).encode("utf-8")).hexdigest()
    random.Random(seed).shuffle(order)
    item["options"] = {chr(ord("A") + i): texts[j] for i, j in enumerate(order)}
    item["answer"] = sorted(chr(ord("A") + i) for i, j in enumerate(order) if texts[j] in correct_texts)


def collect(pilot):
    out = []
    for directory in SEG_DIRS:
        for path in sorted(directory.glob("**/*_segment_*_annotation.json")):
            if pilot and pilot not in path.name:
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if record.get("dense_annotations") and record.get("qa_pairs"):
                out.append((path, record))
    return out


def target_dir(work, source: Path) -> Path:
    tail = source.parts[source.parts.index("annotations_60s") + 1:]
    return work.joinpath("annotations_60s", *tail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--expand-options", action="store_true")
    parser.add_argument("--more-questions", action="store_true")
    parser.add_argument("--pilot")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rate", type=float, default=QUOTA_PER_SECOND)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not (args.expand_options or args.more_questions):
        raise SystemExit("pass --expand-options and/or --more-questions")

    gen = load_generator()
    records = collect(args.pilot)
    calls = len(records) * ((2 if args.expand_options else 0) + (1 if args.more_questions else 0))
    print(f"segments in scope: {len(records)}")
    print(f"batched calls: {calls}  (~{calls / args.rate / 3600:.1f} h at {args.rate}/s)")
    if args.dry_run:
        return 0

    token = os.environ.get("FLOODGATE_PROJECT_TOKEN")
    if not token:
        raise SystemExit("FLOODGATE_PROJECT_TOKEN is required")

    limiter = RateLimiter(args.rate)
    stats = Counter()
    lock = threading.Lock()
    local = threading.local()

    def call(prompt, validator, max_tokens):
        if not hasattr(local, "api"):
            local.api = gen.Floodgate(token)   # requests.Session is not thread-safe
            local.api.session = PacedSession(local.api.session, limiter)
        return gen.generate_valid_json(local.api, prompt, validator, max_tokens)

    def bump(key, amount=1):
        with lock:
            stats[key] += amount

    def process(number, path, record):
        out_path = target_dir(args.work, path)
        if out_path.exists():
            try:
                if json.loads(out_path.read_text(encoding="utf-8")).get("expansion_complete"):
                    bump("segments_skipped")
                    return
            except Exception:
                pass
        evidence = json.dumps(record["dense_annotations"], ensure_ascii=False)

        if args.expand_options:
            targets = {}
            for index, item in enumerate(record["qa_pairs"], 1):
                need = shortfall_of(item)
                if need:
                    targets[f"q{index}"] = (item, need)
                elif isinstance(item.get("options"), dict):
                    bump("questions_already_wide")
            if targets:
                listing = "\n\n".join(
                    f"{qid} (needs {need} new options)\nQUESTION: {item['question']}\n"
                    + "EXISTING OPTIONS (do not repeat or paraphrase):\n"
                    + "\n".join(f"- {item['options'][k]}" for k in sorted(item["options"]))
                    for qid, (item, need) in targets.items()
                )
                try:
                    proposed = call(
                        DISTRACTOR_PROMPT.format(evidence=evidence, questions=listing),
                        make_distractor_validator({q: n for q, (_, n) in targets.items()}),
                        4096)["new_options"]
                except Exception as exc:
                    proposed = None
                    bump("distractor_call_failed")
                    print(f"    distractor call failed {path.name}: {type(exc).__name__}", flush=True)

                if proposed:
                    flat = [(qid, text) for qid in targets for text in proposed[qid]]
                    bump("distractors_proposed", len(flat))
                    numbered = "\n".join(f"{i + 1}. {t}" for i, (_, t) in enumerate(flat))
                    try:
                        verdicts = call(
                            VERIFY_PROMPT.format(evidence=evidence, statements=numbered),
                            make_verdict_validator(len(flat)), 2048)["verdicts"]
                    except Exception as exc:
                        # Never promote an unverified distractor.
                        verdicts = None
                        bump("verify_call_failed")
                        print(f"    verify call failed {path.name}: {type(exc).__name__}", flush=True)

                    if verdicts:
                        accepted = {qid: [] for qid in targets}
                        for (qid, text), verdict in zip(flat, verdicts):
                            bump(f"verdict_{verdict}")
                            if verdict == "contradicted":
                                accepted[qid].append(text)
                        retry = {qid: need for qid, (item, need) in targets.items() if not accepted[qid]}
                        if retry:
                            # All candidates were rejected; ask once more before giving up.
                            again = "\n\n".join(
                                f"{qid} (needs {need} new options)\nQUESTION: {targets[qid][0]['question']}\n"
                                + "EXISTING OPTIONS (do not repeat or paraphrase):\n"
                                + "\n".join(f"- {targets[qid][0]['options'][k]}" for k in sorted(targets[qid][0]["options"]))
                                for qid, need in retry.items())
                            try:
                                more = call(DISTRACTOR_PROMPT.format(evidence=evidence, questions=again),
                                            make_distractor_validator(retry), 4096)["new_options"]
                                flat2 = [(qid, t) for qid in retry for t in more[qid]]
                                bump("distractors_proposed", len(flat2))
                                v2 = call(VERIFY_PROMPT.format(
                                    evidence=evidence,
                                    statements="\n".join(f"{i + 1}. {t}" for i, (_, t) in enumerate(flat2))),
                                    make_verdict_validator(len(flat2)), 2048)["verdicts"]
                                for (qid, text), verdict in zip(flat2, v2):
                                    bump(f"verdict_{verdict}")
                                    if verdict == "contradicted":
                                        accepted[qid].append(text)
                                bump("second_round_used")
                            except Exception as exc:
                                bump("second_round_failed")
                                print(f"    second widen round failed {path.name}: {type(exc).__name__}", flush=True)
                        for qid, (item, _) in targets.items():
                            if not accepted[qid]:
                                bump("questions_not_widened")
                                continue
                            keys = sorted(item["options"])
                            correct = {item["options"][k] for k in item["answer"]}
                            texts = [item["options"][k] for k in keys] + accepted[qid]
                            reletter(item, texts, correct, "|expanded")
                            item["option_count_expanded_to"] = len(texts)
                            bump("questions_widened")
                            bump("distractors_added", len(accepted[qid]))

        if args.more_questions:
            count = NEW_QUESTIONS_PER_SEGMENT
            asked = "\n".join(f"- {q['question']}" for q in record["qa_pairs"])
            try:
                extra = call(
                    QUESTION_PROMPT.format(count=count, evidence=evidence, existing=asked,
                                           n_single=count - count // 2,
                                           target_single=TARGET_SINGLE, target_multi=TARGET_MULTI),
                    make_question_validator(count), 8192)
                # New questions come back with the correct option written first, like the
                # originals did; shuffle them too or the 61% blind baseline returns.
                gen.shuffle_options(extra["qa_pairs"])
                for item in extra["qa_pairs"]:
                    item["generated_by"] = "expand_mcq/more-questions"
                    item["new_question_shuffled"] = True
                record["qa_pairs"].extend(extra["qa_pairs"])
                bump("questions_generated", len(extra["qa_pairs"]))
            except Exception as exc:
                bump("question_call_failed")
                print(f"    question generation failed {path.name}: {type(exc).__name__}", flush=True)

        record["option_order"] = "shuffled"
        record["mcq_schema_version"] = "runningbench-mcq-v4-expanded"
        record["expanded_from"] = str(path)
        record["expansion_complete"] = True
        record["expansion_spec"] = {
            "target_single": TARGET_SINGLE, "target_multi": TARGET_MULTI,
            "added_questions_per_segment": NEW_QUESTIONS_PER_SEGMENT if args.more_questions else 0,
        }
        gen.write_json(out_path, record)
        bump("segments_written")
        print(f"[{number}/{len(records)}] {path.name} -> {len(record['qa_pairs'])} questions", flush=True)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, n, p, r): p for n, (p, r) in enumerate(records, 1)}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                bump("segments_failed")     # one bad segment must not end the run
                print(f"    segment failed {futures[future].name}: {type(exc).__name__}: {exc}", flush=True)

    print(f"\nelapsed {(time.monotonic() - started) / 60:.1f} min")
    print(json.dumps(dict(sorted(stats.items())), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
