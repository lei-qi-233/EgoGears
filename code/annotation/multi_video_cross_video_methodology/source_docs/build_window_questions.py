#!/usr/bin/env python3
"""Benchmark questions from the 180-second window annotations.

Why this layer. The 60s segments hold one phase and no turn, so the only questions they
support are "what is in the frame" -- a blind model beats 48.6% of them. Whole videos
support route questions but run 10-33 minutes. The 180s windows sit where structure
starts: the excerpt study measured yield rising from 42% at 1 minute to 60% at 4, and
three minutes is inside that band. Each window already carries a description of its own
trajectory, turns and environment shifts, so the expensive viewing pass is already paid
for; only the questions and the gates cost quota.

What this layer is NOT. These annotations are aggregations of three 60s captions --
their `derived_from` field names them -- not independent viewings. Every question written
from them inherits whatever the 60s pass got wrong, which is exactly what the 480p
re-check gate exists to catch. Treat an ungated question from here as unverified.

Format. Eight options, three correct. Measured, not assumed: on the segment corpus a
blind model scores 64.7% against single-answer questions and 25.3% against multi-answer
ones, and on the full-video corpus 8-option multi runs 7.3%. Single-answer questions
leak two ways at once -- picking the longest option alone wins 42.2% of them -- and
repairing one leak leaves the other, so the format is chosen up front rather than fixed
later.

Gates, in this order:

  1. shuffle        deterministic, seeded on the question text
  2. structure      arity, distinct options, no empty text
  3. leak scan      filename/speed/"per the description" blacklist
  4. blind guess x3 answer with no video; >=2 hits of 3 drops the question
  5. visual recheck 480p clip of that exact window; only "supported" survives

Order is not arbitrary. Shuffling must precede the blind test: generators favour putting
the answer first and blind models favour picking first, and testing before the shuffle
once produced a 56.6% guess rate that was an artifact of position, not leakage. The blind
test must precede the re-check because a question that fails it needs no video call --
the most expensive step in the pipeline.

Usage:
  build_window_questions.py --pilot 20        # 20 windows, full gates, reports CIs
  build_window_questions.py --all --jobs 2
"""
import argparse
import base64
import collections
import concurrent.futures
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import threading
from pathlib import Path

BASE = Path("/mnt/task_runtime/bolt/gdrive_relay")
SRC = "/mnt/data/data_anno/runningbench_gap_repair/annotations_180s"
CLIPS = Path("/mnt/data/data_anno/runningbench_windows/clips_480p")
OUT = Path("/mnt/data/data_anno/runningbench_windows")
QUOTA_PER_SECOND = 0.45
SEED = 20260831

TYPES = ("window-event-order", "window-turn-sequence",
         "window-environment-shift", "window-object-timing")

# Words that would let a reader answer from the question text or from artifacts of how
# the corpus was built, rather than from the video.
# Kept deliberately narrow. These are corpus artifacts -- filename fragments, and phrases
# that point at the written annotation instead of the footage. Words that merely describe
# what is on screen do not belong here: an earlier version blacklisted "segment", which
# rejected the entirely fair phrasing "the video segment from 06:00 to 09:00" and burned a
# retry on every occurrence. A leak filter that fights natural English costs quota and
# teaches the generator nothing.
LEAK = ("trial", "traj", ".mov", ".mp4", "filename", "annotation",
        "described", "the description", "the evidence", "per the", "according to the")

# Scenery this corpus does not contain. A distractor naming one of these is not a
# distractor -- anyone who knows the setting drops it without watching. This list is the
# defence the segment corpus never had, and its absence is why options there offered
# engine noise and dashboard views for a question about someone running.
OUT_OF_SCENE = ("beach", "highway", "motorway", "forest", "woods", "indoor", "shopping mall",
                "airport terminal", "subway", "train station", "stadium", "boardwalk",
                "desert", "mountain trail", "swimming", "driving", "dashboard", "cockpit")

Q_PROMPT = """Write exactly {n} benchmark questions about ONE 180-second stretch of a first-person
walking/running video, anonymous id {vid}. The stretch runs from {t0} to {t1} of that recording.

EVIDENCE — everything known about this stretch:
{evidence}

Each question must need the WHOLE three minutes. A question answerable from any single
frame, or from any one minute of it, does not belong here: ask about order, about change
across the stretch, about what happened before or after what.

Use these types, spread evenly: {types}
  window-event-order        the sequence in which events occur across the stretch
  window-turn-sequence      the trajectory: which turns, in which order, how sharp
  window-environment-shift  how the surroundings change from the start to the end
  window-object-timing      which objects appear in which part of the stretch

FORMAT — exactly 8 options, of which exactly 3 are correct. Return the three correct ones
first; they get shuffled afterwards.

THE FIVE WRONG OPTIONS ARE THE HARD PART. Read all of this.

A wrong option must be wrong ONLY because the video says otherwise — never because a
reader can rule it out from the setting. Someone who knows this is a person on foot
outdoors, and nothing else, must find all eight equally possible. So:

- Build them from THIS stretch's own vocabulary: its objects, its paths, its turns. A
  wrong option that names something absent from the evidence — a beach, a dashboard, an
  indoor hall — is worthless.
- Make them wrong by COMPOSITION, not by content: the right things in the wrong order,
  an event placed in the wrong third, two objects swapped, a turn given the wrong
  direction or the wrong sharpness.
- Do NOT write each wrong option as a correct one with a single detail edited. If every
  wrong option is one edit from a right one, then on each detail the correct value is
  the majority value, and the whole answer can be recovered by taking the most common
  colour, the most common count, the most common side — no video needed. This is the
  single most common way a question set leaks; a set built that way once drove blind
  guessing from 48.6% to 75%.
- Vary which details differ and how many: change two or three at once, in different
  combinations, so no option sits at the centre of the set.
- Keep all eight the same length and the same specificity. The correct options must not
  be the longest or the most detailed — picking the longest option alone wins 42.2% of
  single-answer questions in an earlier corpus, and the same habit leaks here.

Every question must also carry evidence_spans: the seconds within the FULL recording
(not within this stretch) that show the answer, so the claim can be re-checked.

Return only:
{{"questions": [{{"question": "...", "question_type": "...",
  "options": ["correct 1", "correct 2", "correct 3", "wrong 1", "wrong 2", "wrong 3", "wrong 4", "wrong 5"],
  "evidence_spans": [{{"start_seconds": 0, "end_seconds": 0, "description": "..."}}],
  "why_hard": "what a viewer must track across the three minutes to answer"}}]}}"""

RECHECK_PROMPT = """Verify one benchmark question against the clip, which is exactly the stretch the
question is about.

QUESTION: {question}
OPTIONS: {options}
MARKED CORRECT: {answer}

Watch the clip. Decide whether the marked options — all of them, and no others — are the
correct answer.

- "supported"    the marked options are right and the unmarked ones are wrong
- "contradicted" the clip shows something else; say what in notes
- "insufficient" the clip does not settle it

Judge only what the clip shows. If a detail is legible, it counts, whatever the encode.

Return only: {{"verdict": "...", "correct_answer_if_different": null, "notes": "..."}}"""


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * (c - h) / d, 100 * (c + h) / d)


INLINE_CAP = 11 * 1024 * 1024


def video_part(path):
    """Inline one clip. The API rejects a request whose inline payload exceeds 11 MiB."""
    data = Path(path).read_bytes()
    if len(data) > INLINE_CAP:
        raise ValueError(f"clip is {len(data)/2**20:.1f} MiB, over the {INLINE_CAP/2**20:.0f} MiB inline cap")
    return {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(data).decode("ascii")}}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mmss(sec):
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def scene_vocabulary(evidence):
    return set(re.findall(r"[a-z]{4,}", json.dumps(evidence, ensure_ascii=False).lower()))


def make_validator(n, t0, t1, vocab):
    def check(d):
        qs = d.get("questions")
        if not isinstance(qs, list) or len(qs) != n:
            raise ValueError(f"need exactly {n} questions, got {len(qs) if isinstance(qs, list) else '?'}")
        for q in qs:
            stem = (q.get("question") or "").strip()
            if not stem:
                raise ValueError("empty question")
            low = stem.lower()
            for bad in LEAK:
                if bad in low:
                    raise ValueError(f"question leaks {bad!r}: {stem[:60]}")
            if q.get("question_type") not in TYPES:
                raise ValueError(f"bad question_type {q.get('question_type')!r}")
            opts = q.get("options")
            if not isinstance(opts, list) or len(opts) != 8:
                raise ValueError(f"need exactly 8 options, got {len(opts) if isinstance(opts, list) else '?'}")
            seen = set()
            for o in opts:
                if not isinstance(o, str) or not o.strip():
                    raise ValueError("options must be non-empty strings")
                if "\n" in o:
                    raise ValueError("one option per string; this one contains a newline")
                key = o.strip().lower()
                if key in seen:
                    raise ValueError(f"duplicate option: {o[:40]!r}")
                seen.add(key)
                for bad in OUT_OF_SCENE:
                    if bad in key:
                        raise ValueError(f"option names out-of-scene {bad!r}: {o[:50]}")
            # The correct three are returned first; they must not be the longest three.
            lengths = [len(o) for o in opts]
            if sorted(range(8), key=lambda i: -lengths[i])[:3] == [0, 1, 2]:
                raise ValueError("the three correct options are the three longest")
            spans = q.get("evidence_spans")
            if not isinstance(spans, list) or not spans:
                raise ValueError("evidence_spans required")
            for s in spans:
                a, b = s.get("start_seconds"), s.get("end_seconds")
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or b <= a:
                    raise ValueError(f"bad span {s}")
                if b < t0 - 5 or a > t1 + 5:
                    raise ValueError(f"span {a}-{b} outside window {t0}-{t1}")
    return check


def validate_verdict(d):
    if d.get("verdict") not in ("supported", "contradicted", "insufficient"):
        raise ValueError(f"bad verdict {d.get('verdict')!r}")


def shuffle(q):
    texts = q["options"]
    correct = set(range(3))                      # generator returns the correct three first
    order = list(range(len(texts)))
    random.Random(hashlib.sha256(q["question"].encode("utf-8")).hexdigest()).shuffle(order)
    q["options"] = {chr(ord("A") + i): texts[src] for i, src in enumerate(order)}
    q["answer"] = sorted(chr(ord("A") + i) for i, src in enumerate(order) if src in correct)
    q["option_order"] = "shuffled"


def collect():
    rows = []
    for path in sorted(glob.glob(f"{SRC}/**/*.json", recursive=True)):
        rec = json.load(open(path))
        source = rec.get("original_video_path") or ""
        stem = os.path.basename(source).rsplit(".", 1)[0]
        idx = int(rec.get("chunk_index", 0))
        clip = CLIPS / stem / f"{stem}_w{idx}.480p.mp4"
        rows.append({"file": path, "video": stem, "chunk": idx,
                     "t0": float(rec["start_time_sec"]), "t1": float(rec["end_time_sec"]),
                     "evidence": rec.get("dense_annotations"),
                     "open_qa": rec.get("qa_pairs") or [],
                     "clip": str(clip) if clip.exists() else None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="how many windows to run")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-per-window", type=int, default=4)
    # The distractor repair is usually holding most of the 0.45 req/s quota; two workers
    # here share it rather than starving both runs into 429 backoff.
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=str(OUT / "questions.json"))
    a = ap.parse_args()
    if not (a.pilot or a.all):
        ap.error("pass --pilot N or --all")

    fv = load("fv", BASE / "build_fullvideo_annotations.py")
    gen = load("gen", BASE / "repair_runningbench_annotations.py")

    rows = collect()
    print(f"{len(rows)} windows; {sum(1 for r in rows if r['clip'])} have a 480p clip", flush=True)
    if a.pilot:
        rng = random.Random(SEED)
        by_video = collections.defaultdict(list)
        for r in rows:
            by_video[r["video"]].append(r)
        picked = []
        for vid in sorted(by_video):
            group = by_video[vid]
            rng.shuffle(group)
            picked += group[:max(1, round(a.pilot / len(by_video)))]
        rng.shuffle(picked)
        rows = picked[:a.pilot]
        print(f"pilot: {len(rows)} windows from {len({r['video'] for r in rows})} videos", flush=True)

    api = gen.Floodgate("")
    api.session = fv.PacedSession(api.session, fv.RateLimiter(QUOTA_PER_SECOND))
    OUT.mkdir(parents=True, exist_ok=True)

    done = {}
    if os.path.exists(a.out):
        try:
            prev = json.load(open(a.out))
            done = {(q["video"], q["chunk"]) for q in prev.get("questions", [])}
            questions = prev.get("questions", [])
            print(f"resuming: {len(questions)} questions over {len(done)} windows", flush=True)
        except Exception:
            questions = []
    else:
        questions = []
    rows = [r for r in rows if (r["video"], r["chunk"]) not in done]
    print(f"to do: {len(rows)} windows", flush=True)

    lock = threading.Lock()
    counter = {"n": 0}

    def work(r):
        vocab = scene_vocabulary(r["evidence"])
        try:
            data = gen.generate_valid_json(
                api,
                Q_PROMPT.format(n=a.n_per_window, vid=r["video"], t0=mmss(r["t0"]), t1=mmss(r["t1"]),
                                evidence=json.dumps(r["evidence"], ensure_ascii=False),
                                types=", ".join(TYPES)),
                make_validator(a.n_per_window, r["t0"], r["t1"], vocab), 32768)
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}"

        out = []
        for q in data["questions"]:
            shuffle(q)
            q.update({"video": r["video"], "chunk": r["chunk"], "window_sec": [r["t0"], r["t1"]],
                      "clip": r["clip"]})
            gates = {"structure": "pass", "leak_scan": "pass"}

            arity = "Exactly three options are correct."
            hits, guesses = 0, []
            for t in range(3):
                try:
                    bg = gen.generate_valid_json(
                        api, fv.BLIND_PROMPT.format(question=q["question"],
                                                    options=json.dumps(q["options"], ensure_ascii=False),
                                                    arity=f"{arity} (attempt {t + 1})"),
                        fv.validate_blind, 4096)
                    g = sorted(bg["answer"])
                except Exception:
                    g = ["?"]
                guesses.append(g)
                hits += g == sorted(q["answer"])
            gates["blind_guess"] = {"guesses": guesses, "hits_of_3": hits, "matches_gold": hits >= 2}

            if gates["blind_guess"]["matches_gold"]:
                # Guessable without video; the re-check is the most expensive call in the
                # pipeline and would only confirm a question already lost.
                gates["visual_recheck"] = {"verdict": "skipped", "notes": "blind-guessable"}
            elif not r["clip"]:
                gates["visual_recheck"] = {"verdict": "skipped", "notes": "no clip on disk"}
            else:
                try:
                    parts = [{"text": RECHECK_PROMPT.format(
                                  question=q["question"],
                                  options=json.dumps(q["options"], ensure_ascii=False),
                                  answer=json.dumps(q["answer"]))},
                             video_part(r["clip"])]
                    gates["visual_recheck"] = gen.clean_json(
                        api.generate(gen.CAPTION_MODEL, parts, 8192, json_mode=True))
                    validate_verdict(gates["visual_recheck"])
                except Exception as exc:
                    gates["visual_recheck"] = {"verdict": "error", "notes": type(exc).__name__}
            q["gates"] = gates
            out.append(q)
        return out, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for got, err in pool.map(work, rows):
            with lock:
                counter["n"] += 1
                i = counter["n"]
                if err:
                    print(f"  !! window {i}: {err}", flush=True)
                questions += got
                usable = sum(1 for q in questions
                             if q["gates"].get("visual_recheck", {}).get("verdict") == "supported"
                             and not q["gates"]["blind_guess"].get("matches_gold"))
                print(f"  {i}/{len(rows)} windows · {len(questions)} questions · {usable} usable", flush=True)
                json.dump({"n": len(questions), "questions": questions},
                          open(a.out, "w"), ensure_ascii=False, indent=1)

    total = len(questions)
    usable = [q for q in questions
              if q["gates"].get("visual_recheck", {}).get("verdict") == "supported"
              and not q["gates"]["blind_guess"].get("matches_gold")]
    blind = sum(1 for q in questions if q["gates"]["blind_guess"].get("matches_gold"))
    print(f"\n=== {total} questions from {len(rows)} windows -> {a.out}")
    if total:
        lo, hi = wilson(len(usable), total)
        bl, bh = wilson(blind, total)
        print(f"usable        {len(usable)}/{total} = {100*len(usable)/total:.1f}%  95% CI [{lo:.1f}, {hi:.1f}]")
        print(f"blind-guessed {blind}/{total} = {100*blind/total:.1f}%  95% CI [{bl:.1f}, {bh:.1f}]")
        print(f"  (segment corpus 48.6%, full-video 8-option multi 7.3%)")
        by_v = collections.Counter()
        for q in questions:
            v = q["gates"].get("visual_recheck", {})
            by_v[v.get("verdict") if isinstance(v, dict) else v] += 1
        print(f"recheck       {dict(by_v)}")
        by_t = collections.defaultdict(collections.Counter)
        for q in questions:
            ok = q in usable
            by_t[q["question_type"]]["ok" if ok else "no"] += 1
        for t, c in sorted(by_t.items()):
            n = c["ok"] + c["no"]
            print(f"  {t:26s} {c['ok']:3d}/{n:3d}  {100*c['ok']/n:3.0f}%")
        if total < 100:
            print(f"\nNOTE: n={total} is small. Distinguishing a ~7-point difference in guess rate "
                  f"needs about 800 per group; read these intervals, not the point estimates.")


if __name__ == "__main__":
    sys.exit(main())
