#!/usr/bin/env python3
"""Rebuild distractors for blind-guessable segment questions, then measure whether it worked.

Design note -- why there is a control arm.

Repair candidates are selected *because* the gate caught them being guessed. Measuring
them once more after repair would confound the repair with regression to the mean: a
question guessed once at temperature 0.2 is not certainly guessable, and re-measuring a
set selected on a noisy positive will drift down on its own. So every question is blind-
tested twice -- once on its ORIGINAL options (control) and once on the REPAIRED options
(treatment) -- in the same run, same model, same prompt. The control arm carries the
regression, so the control-vs-treatment gap is the repair effect.

This is the trap the project already fell into once: the near-miss rewrite drove the
metric it targeted ("correct option is longest": 48% -> 28%) while blind-guessability
went 48.6% -> 75%. Only an end-to-end blind measurement catches that, never a proxy.

Prompts are copied verbatim from repair_segment_distractors.py so the rewrite rule under
test is the project's current one (change two or more details, no detail of the correct
option in more than half the set, distractors near-misses of each other).
"""
import argparse, collections, hashlib, json, math, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate_segments_60s import (BLIND_PROMPT, Floodgate, RateLimiter, parse, wilson)

QUOTA_PER_SECOND = 0.45
SEED = 20260831

REPAIR_PROMPT = """Rewrite the WRONG options of one benchmark question about a 60-second first-person
walking/running video. The correct option is fixed — never change, paraphrase or restate it.

EVIDENCE (the only ground truth):
{evidence}

QUESTION: {question}
CORRECT OPTION(S), verbatim and unchangeable:
{correct}

Write exactly {n} wrong options, and treat the whole SET as the thing you are designing.

The trap to avoid, above everything else: if every wrong option is the correct one with a
single detail edited, then on each detail the correct value is the majority value, and the
answer can be recovered by taking the most common colour, the most common count, the most
common side — no video needed. A set like this is worthless:

    dark grey  / multi-story / glass        light yellow / single-story / glass
    light yellow / multi-story / glass  <-- correct, and the majority on every axis
    dark red   / multi-story / glass        light yellow / multi-story / wooden

So:
- Change TWO OR MORE details at once in most wrong options, and change different combinations
  in different options. No detail of the correct option may appear in more than half the set.
- Make the wrong options near-misses OF EACH OTHER too, not only of the correct one. A reader
  who cannot see the video must find no option more "central" or more self-consistent than
  the others.
- Stay inside the world of this video. A wrong option that a person rules out just by knowing
  this is a first-person walking/running recording is useless: no dashboards, no engine noise,
  no indoor scenes unless the evidence has them.
- Be CONTRADICTED by the evidence: it must positively rule the option out. Not merely
  unmentioned, not partly true.
- Match the correct option's specificity and length: between {lo} and {hi} characters, and the
  correct option must not be the longest or the most detailed one in the set.

Return only: {{"options": ["...", "..."]}}

Each option is ONE plain string in that array. Do not prefix them with "-" or a bullet,
do not put a newline inside one, and do not pack several options into a single string —
returning the whole list as one item is the most common way this call fails."""

VERIFY_PROMPT = """Judge each numbered statement against the evidence from a first-person video segment.

EVIDENCE:
{evidence}

STATEMENTS:
{statements}

Answer strictly per statement:
- "contradicted" if the evidence positively rules it out;
- "supported" if the evidence indicates it is true;
- "unclear" if the evidence neither confirms nor rules it out.

Return only: {{"verdicts": ["contradicted", "unclear", ...]}}"""


def clean_option(o):
    return re.sub(r"^\s*[-*•]\s+", "", o).strip() if isinstance(o, str) else o


def validate_repair(d, n, lo, hi, gold):
    opts = d.get("options")
    if not isinstance(opts, list) or len(opts) != n:
        raise ValueError(f"need exactly {n} options, got {len(opts) if isinstance(opts, list) else '?'}")
    d["options"] = opts = [clean_option(o) for o in opts]
    seen = set()
    for o in opts:
        if not isinstance(o, str) or not o.strip():
            raise ValueError("options must be non-empty strings")
        if "\n" in o:
            raise ValueError("one option per string; this one contains a newline")
        k = o.strip().lower()
        if k in gold:
            raise ValueError("a wrong option repeats the correct option")
        if k in seen:
            raise ValueError("duplicate options")
        seen.add(k)
        if not (lo * 0.8 <= len(o) <= hi * 1.25):
            raise ValueError(f"option length {len(o)} outside {int(lo)}..{int(hi)}")
    return opts


def json_call(api, prompt, validator, max_tokens=8192, rounds=5):
    """Generate until the payload validates; the validator's message is fed back as repair advice."""
    last = None
    for _ in range(rounds):
        raw = api.generate(prompt if last is None else
                           prompt + f"\n\nYour previous answer was rejected: {last}\nFix exactly that.",
                           max_tokens)
        try:
            d = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                last = "not JSON"
                continue
            try:
                d = json.loads(m.group(0))
            except Exception:
                last = "not JSON"
                continue
        try:
            return validator(d)
        except Exception as exc:
            last = str(exc)
    raise RuntimeError(f"validation failed after {rounds} rounds: {last}")


def shuffle_letters(texts, correct_idx, seed_text):
    """Deterministic reshuffle, matching repair_segment_distractors.shuffle."""
    order = list(range(len(texts)))
    random.Random(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()).shuffle(order)
    options = {chr(ord("A") + i): texts[src] for i, src in enumerate(order)}
    answer = sorted(chr(ord("A") + i) for i, src in enumerate(order) if src in correct_idx)
    return options, answer


def blind(api, question, options, gold, votes=3):
    """Blind-test a question. Raises if the API call fails -- never treats that as a pass.

    The earlier version swallowed the exception and scored the missing answer as a non-guess,
    which reads identically to "the model could not guess it". After a 429 burst poisoned a
    long-running session, every later call raised, and 243 questions were recorded as sound
    when they had never actually been measured -- 8 of 10 spot-checked were in fact guessable.
    A failed measurement is missing data, not evidence of quality, so it propagates and the
    caller records an error.

    A response that arrives but does not parse is different: the model answered unusably, and
    that genuinely is a non-guess.
    """
    arity = ("Exactly one option is correct." if len(gold) == 1
             else "More than one option is correct.")
    hits, guesses = 0, []
    for t in range(votes):
        raw = api.generate(BLIND_PROMPT.format(
            question=question, options=json.dumps(options, ensure_ascii=False),
            arity=arity + f" (attempt {t+1})"))
        g = parse(raw, options)
        guesses.append(g)
        hits += (g == sorted(gold))
    return hits, guesses


def mcnemar(b, c):
    """Exact-ish McNemar on discordant pairs (b: control-only, c: treatment-only)."""
    n = b + c
    if n == 0:
        return 1.0
    chi = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--gate", default="segments_60s_gate.progress.jsonl")
    ap.add_argument("--n", type=int, default=0, help="0 = every blind-guessable question found so far")
    # One vote per arm is enough here. At temperature 0.2 the three votes agree on 86.5% of
    # questions, so the extra two rarely change a verdict, and this is a population-level
    # paired comparison over hundreds of items -- single-draw noise averages out across the
    # sample instead of having to be suppressed per question, as a per-question gate needs.
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--verify-every", type=int, default=5,
                    help="run the contradicted-check on every Nth question (0 = never). "
                         "It audits distractor quality; it is not part of the paired measurement.")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="repair_measure.json")
    ap.add_argument("--checkpoint", default="repair_measure.progress.jsonl")
    a = ap.parse_args()

    qs = {q["id"]: q for q in (json.loads(l) for l in open(f"{a.qdir}/questions/segments_60s.jsonl"))}
    caps = {}
    for l in open(f"{a.qdir}/captions/segments_60s.jsonl"):
        c = json.loads(l)
        caps[(c["video"], tuple(c["span_sec"]))] = c["caption"]

    cand = []
    for l in open(a.gate):
        r = json.loads(l)
        if r["blind_guessable"] and r["id"] in qs:
            cand.append(qs[r["id"]])
    rng = random.Random(SEED)
    rng.shuffle(cand)
    if a.n:
        cand = cand[:a.n]
    print(f"repair candidates: {len(cand)}", flush=True)

    done = {}
    cp = Path(a.checkpoint)
    if cp.exists():
        for l in cp.open():
            try:
                r = json.loads(l)
                done[r["id"]] = r
            except Exception:
                pass
        print(f"resuming: {len(done)} done", flush=True)

    api = Floodgate(os.environ.get("FLOODGATE_PROJECT_TOKEN", ""), RateLimiter(QUOTA_PER_SECOND))
    fh = cp.open("a")
    lock = threading.Lock()
    todo = [q for q in cand if q["id"] not in done]
    t0, counter = time.time(), {"n": 0}

    def work(q):
        rec = {"id": q["id"], "arity": q.get("arity"), "n_options": q.get("n_options"),
               "votes": a.votes}
        try:
            opts, ans = q["options"], sorted(q["answer"])
            gold_texts = [opts[k] for k in ans]
            wrong_texts = [v for k, v in sorted(opts.items()) if k not in ans]
            ev = caps.get((q.get("video"), tuple(q.get("span_sec") or [])))
            if ev is None:
                raise RuntimeError("no evidence")
            clen = [len(t) for t in gold_texts]
            lo, hi = min(clen) * 0.75, max(clen) * 1.05
            gold_norm = {t.strip().lower() for t in gold_texts}

            # --- control: the ORIGINAL question, measured again in this same run ---
            c_hits, _ = blind(api, q["question"], opts, ans, a.votes)

            # --- treatment: rebuild the distractors, then measure ---
            new_wrong = json_call(api, REPAIR_PROMPT.format(
                evidence=json.dumps(ev, ensure_ascii=False), question=q["question"],
                correct="\n".join(gold_texts), n=len(wrong_texts), lo=int(lo), hi=int(hi)),
                lambda d: validate_repair(d, len(wrong_texts), lo, hi, gold_norm))

            texts = gold_texts + new_wrong
            new_opts, new_ans = shuffle_letters(texts, set(range(len(gold_texts))), q["question"])
            t_hits, _ = blind(api, q["question"], new_opts, new_ans, a.votes)

            # --- audit only: does the evidence actually rule the new distractors out? ---
            verdicts = None
            with lock:
                counter["seen"] = counter.get("seen", 0) + 1
                do_verify = a.verify_every and counter["seen"] % a.verify_every == 0
            if do_verify:
                verdicts = json_call(api, VERIFY_PROMPT.format(
                    evidence=json.dumps(ev, ensure_ascii=False),
                    statements="\n".join(f"{i+1}. {t}" for i, t in enumerate(new_wrong))),
                    lambda d: (d["verdicts"] if isinstance(d.get("verdicts"), list)
                               and len(d["verdicts"]) == len(new_wrong)
                               else (_ for _ in ()).throw(ValueError("bad verdicts"))), 4096)

            need = max(1, (a.votes + 1) // 2)   # 1 of 1, or 2 of 3
            rec.update({"control_hits": c_hits, "control_guessable": c_hits >= need,
                        "treat_hits": t_hits, "treat_guessable": t_hits >= need,
                        "n_rebuilt": len(new_wrong),
                        "contradicted": (sum(1 for v in verdicts if v == "contradicted")
                                         if verdicts else None),
                        "verdicts": verdicts, "new_options": new_opts, "new_answer": new_ans,
                        "error": None})
        except Exception as exc:
            rec.update({"error": f"{type(exc).__name__}: {exc}"})
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[q["id"]] = rec
            counter["n"] += 1
            i = counter["n"]
            if i % 25 == 0:
                ok = [r for r in done.values() if not r.get("error")]
                if ok:
                    cg = sum(1 for r in ok if r["control_guessable"])
                    tg = sum(1 for r in ok if r["treat_guessable"])
                    el = (time.time() - t0) / 60
                    print(f"  [{i}/{len(todo)}] control {100*cg/len(ok):.1f}%  "
                          f"repaired {100*tg/len(ok):.1f}%  (n={len(ok)})  {el:.0f}m", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(work, todo))
    fh.close()

    ok = [r for r in done.values() if not r.get("error")]
    n = len(ok)
    cg = sum(1 for r in ok if r["control_guessable"])
    tg = sum(1 for r in ok if r["treat_guessable"])
    b = sum(1 for r in ok if r["control_guessable"] and not r["treat_guessable"])
    c = sum(1 for r in ok if r["treat_guessable"] and not r["control_guessable"])
    clo, chi = wilson(cg, n)
    tlo, thi = wilson(tg, n)

    def split(key):
        by = collections.defaultdict(list)
        for r in ok:
            by[r[key]].append(r)
        out = []
        for k, v in sorted(by.items(), key=lambda x: -len(x[1])):
            g1 = sum(1 for r in v if r["control_guessable"])
            g2 = sum(1 for r in v if r["treat_guessable"])
            out.append({str(key): k, "n": len(v),
                        "control_pct": round(100 * g1 / len(v), 1),
                        "repaired_pct": round(100 * g2 / len(v), 1),
                        "delta_pt": round(100 * (g2 - g1) / len(v), 1)})
        return out

    aud = [r for r in ok if r.get("contradicted") is not None]
    reb = sum(r["n_rebuilt"] for r in aud)
    con = sum(r["contradicted"] for r in aud)
    summary = {
        "n_paired": n, "errors": len(done) - n,
        "control_guessable_pct": round(100 * cg / n, 1) if n else None,
        "control_ci95": [round(100 * clo, 1), round(100 * chi, 1)],
        "repaired_guessable_pct": round(100 * tg / n, 1) if n else None,
        "repaired_ci95": [round(100 * tlo, 1), round(100 * thi, 1)],
        "delta_pt": round(100 * (tg - cg) / n, 1) if n else None,
        "discordant_fixed": b, "discordant_broken": c,
        "mcnemar_p": round(mcnemar(b, c), 6),
        "votes_mix": dict(collections.Counter(r.get("votes") for r in ok)),
        "audited_questions": len(aud), "distractors_rebuilt_audited": reb,
        "contradicted_pct": round(100 * con / reb, 1) if reb else None,
        "by_arity": split("arity"), "by_n_options": split("n_options"),
    }
    Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
