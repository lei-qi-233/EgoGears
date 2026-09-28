#!/usr/bin/env python3
"""Measure-then-repair every single-choice segment question, and keep the evidence.

Why single-choice: a 587-question paired experiment (repair_measure.json) found the
distractor rewrite moves single-choice from 95.4% to 73.4% blind-guessable
(-22.0pt, McNemar p<0.0001) while multi-answer does not move at all
(51.6% -> 56.1%, p=0.48). The project's --multi-only default aims the repair at the
half that does not respond; this aims it at the half that does.

One pass does both jobs. A question is blind-tested first and only repaired if the
model actually guesses it, so questions that are already sound are never touched --
in the paired experiment 13 of 432 single-choice questions were made *worse* by an
unconditional rewrite, and this avoids paying that cost on the ones that do not need
it. Expect ~2.4 calls per question rather than the 4 a separate gate-then-repair pass
would spend.

Every question keeps its before/after verdicts, so the output is auditable and a
question whose repair did not help can be revisited without re-measuring.
"""
import argparse, collections, hashlib, json, math, os, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate_segments_60s import BLIND_PROMPT, Floodgate, RateLimiter, parse, wilson
from repair_and_measure import (REPAIR_PROMPT, json_call, shuffle_letters,
                                validate_repair, blind)

QUOTA_PER_SECOND = 0.45
SEED = 20260906


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--seed-from", default="repair_measure.progress.jsonl",
                    help="reuse the paired experiment's already-measured questions")
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="singles_repaired.jsonl")
    ap.add_argument("--summary", default="singles_repaired_summary.json")
    ap.add_argument("--checkpoint", default="singles_repaired.progress.jsonl")
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(f"{a.qdir}/questions/segments_60s.jsonl")]
    singles = [q for q in qs if q.get("arity") == "single"]
    caps = {}
    for l in open(f"{a.qdir}/captions/segments_60s.jsonl"):
        c = json.loads(l)
        caps[(c["video"], tuple(c["span_sec"]))] = c["caption"]
    if a.limit:
        singles = singles[:a.limit]
    print(f"single-choice questions: {len(singles)}", flush=True)

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

    # Fold in the paired experiment: those questions already have a before/after pair
    # measured the same way, so re-running them would buy nothing.
    seeded = 0
    if a.seed_from and Path(a.seed_from).exists() and not done:
        for l in open(a.seed_from):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("error") or r.get("arity") != "single":
                continue
            rec = {"id": r["id"], "n_options": r.get("n_options"),
                   "blind_before": r["control_guessable"],
                   "repaired": True, "blind_after": r["treat_guessable"],
                   "options": r.get("new_options"), "answer": r.get("new_answer"),
                   "source": "paired_experiment", "error": None}
            done[r["id"]] = rec
            seeded += 1
        if seeded:
            with cp.open("a") as fh:
                for r in done.values():
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"seeded {seeded} from the paired experiment", flush=True)

    api = Floodgate(os.environ.get("FLOODGATE_PROJECT_TOKEN", ""), RateLimiter(QUOTA_PER_SECOND))
    fh = cp.open("a")
    lock = threading.Lock()
    todo = [q for q in singles if q["id"] not in done]
    print(f"to do: {len(todo)}", flush=True)
    t0, counter = time.time(), {"n": 0}
    need = max(1, (a.votes + 1) // 2)

    def work(q):
        rec = {"id": q["id"], "n_options": q.get("n_options"), "source": "measured"}
        try:
            opts, ans = q["options"], sorted(q["answer"])
            # --- 1. does the model already beat it without the video? ---
            b_hits, _ = blind(api, q["question"], opts, ans, a.votes)
            before = b_hits >= need
            rec.update({"blind_before": before})
            if not before:
                rec.update({"repaired": False, "blind_after": False,
                            "options": opts, "answer": ans, "error": None})
            else:
                gold_texts = [opts[k] for k in ans]
                wrong_texts = [v for k, v in sorted(opts.items()) if k not in ans]
                ev = caps.get((q.get("video"), tuple(q.get("span_sec") or [])))
                if ev is None:
                    raise RuntimeError("no evidence")
                clen = [len(t) for t in gold_texts]
                lo, hi = min(clen) * 0.75, max(clen) * 1.05
                gold_norm = {t.strip().lower() for t in gold_texts}
                # --- 2. rebuild the distractors ---
                new_wrong = json_call(api, REPAIR_PROMPT.format(
                    evidence=json.dumps(ev, ensure_ascii=False), question=q["question"],
                    correct="\n".join(gold_texts), n=len(wrong_texts),
                    lo=int(lo), hi=int(hi)),
                    lambda d: validate_repair(d, len(wrong_texts), lo, hi, gold_norm))
                texts = gold_texts + new_wrong
                new_opts, new_ans = shuffle_letters(texts, set(range(len(gold_texts))), q["question"])
                # --- 3. did it actually help? ---
                a_hits, _ = blind(api, q["question"], new_opts, new_ans, a.votes)
                rec.update({"repaired": True, "blind_after": a_hits >= need,
                            "options": new_opts, "answer": new_ans, "error": None})
        except Exception as exc:
            rec.update({"error": f"{type(exc).__name__}: {exc}"})
        with lock:
            done[q["id"]] = rec
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            counter["n"] += 1
            i = counter["n"]
            if i % 200 == 0:
                ok = [r for r in done.values() if not r.get("error")]
                bad_after = sum(1 for r in ok if r.get("blind_after"))
                el = (time.time() - t0) / 60
                rate = i / el if el else 0
                print(f"  [{i}/{len(todo)}] corpus still-guessable {100*bad_after/len(ok):.1f}% "
                      f"(n={len(ok)})  {el:.0f}m  {rate:.1f}/min  "
                      f"ETA {(len(todo)-i)/rate/60:.1f}h", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(work, todo))
    fh.close()

    ok = [r for r in done.values() if not r.get("error")]
    n = len(ok)
    before_bad = sum(1 for r in ok if r["blind_before"])
    after_bad = sum(1 for r in ok if r["blind_after"])
    touched = [r for r in ok if r.get("repaired")]
    fixed = sum(1 for r in touched if not r["blind_after"])
    blo, bhi = wilson(before_bad, n)
    alo, ahi = wilson(after_bad, n)

    with open(a.out, "w") as out:
        for r in ok:
            out.write(json.dumps({
                "id": r["id"], "options": r["options"], "answer": r["answer"],
                "n_options": r["n_options"], "repaired": bool(r.get("repaired")),
                "blind_before": r["blind_before"], "blind_after": r["blind_after"],
                "screening_status": ("passed_blind_gate" if not r["blind_after"]
                                     else "still_blind_guessable"),
            }, ensure_ascii=False) + "\n")

    summary = {
        "single_choice_total": len(singles), "processed": n, "errors": len(done) - n,
        "blind_before_pct": round(100 * before_bad / n, 1),
        "blind_before_ci95": [round(100 * blo, 1), round(100 * bhi, 1)],
        "blind_after_pct": round(100 * after_bad / n, 1),
        "blind_after_ci95": [round(100 * alo, 1), round(100 * ahi, 1)],
        "delta_pt": round(100 * (after_bad - before_bad) / n, 1),
        "repaired": len(touched), "repair_succeeded": fixed,
        "repair_success_pct": round(100 * fixed / len(touched), 1) if touched else None,
        "usable_after": n - after_bad,
        "sources": dict(collections.Counter(r.get("source") for r in ok)),
    }
    Path(a.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
