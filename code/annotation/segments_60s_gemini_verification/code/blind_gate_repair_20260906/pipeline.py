#!/usr/bin/env python3
"""One pass over segments_60s: measure, repair only if needed, escalate only if repair fails.

Stage per question, stopping at the first one that clears it:

  1. blind-test the ORIGINAL            -> clear? keep it verbatim, 1 call spent
  2. v1 rewrite + blind-test            -> clear? keep v1
  3. v2 adversarial, up to N rounds     -> clear? keep v2   (rewrite -> verify -> blind)
  4. nothing worked                     -> keep the ORIGINAL, flagged still_guessable

Measured stage-by-stage on 432 single-choice questions: 95.4% blind-guessable originally,
73.4% after v1, 33.1% after v1+v2 (v2 rescues 53.0% of what v1 loses, CI[47.3, 58.5]).

Two things this deliberately does NOT do. It never rewrites a question that the blind
model failed to guess -- v1 applied unconditionally made 13 of 432 sound questions worse.
And it never keeps a rewrite that did not beat the blind test: stage 4 reverts to the
original rather than shipping a rewrite whose only evidence is that it was attempted.

Resumes from every earlier pass: questions already resolved as passed are not re-run, and
neither are ones that already exhausted v2's rounds without clearing.
"""
import argparse, collections, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate_segments_60s import Floodgate, RateLimiter, wilson
from repair_and_measure import (REPAIR_PROMPT, json_call, shuffle_letters,
                                validate_repair, blind)
from repair_v2 import REPAIR_V2, FEEDBACK_TMPL, verify_contradicted

QUOTA_PER_SECOND = 0.45
SEED = 20260906


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--curated", default="runningbench_segments_curated.jsonl")
    ap.add_argument("--v2-rounds", type=int, default=3)
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--require-contradicted", type=float, default=0.8)
    ap.add_argument("--out", default="pipeline_summary.json")
    ap.add_argument("--checkpoint", default="pipeline.progress.jsonl")
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(f"{a.qdir}/questions/segments_60s.jsonl")]
    caps = {}
    for l in open(f"{a.qdir}/captions/segments_60s.jsonl"):
        c = json.loads(l)
        caps[(c["video"], tuple(c["span_sec"]))] = c["caption"]

    # Earlier passes already settled some of the corpus; don't pay for them twice.
    settled = {}
    cur = Path(a.curated)
    if cur.exists():
        for l in cur.open():
            r = json.loads(l)
            if r["screening_status"] == "passed" or r.get("source") == "original_unsalvaged":
                settled[r["id"]] = r
        print(f"already settled by earlier passes: {len(settled)}", flush=True)

    done = {}
    cp = Path(a.checkpoint)
    if cp.exists():
        for l in cp.open():
            try:
                r = json.loads(l)
                done[r["id"]] = r
            except Exception:
                pass
        print(f"resuming: {len(done)}", flush=True)

    todo = [q for q in qs if q["id"] not in settled and q["id"] not in done]
    random.Random(SEED).shuffle(todo)
    if a.limit:
        todo = todo[:a.limit]
    print(f"corpus={len(qs)}  to do={len(todo)}", flush=True)

    api = Floodgate(os.environ.get("FLOODGATE_PROJECT_TOKEN", ""), RateLimiter(QUOTA_PER_SECOND))
    fh = cp.open("a")
    lock = threading.Lock()
    t0, counter = time.time(), {"n": 0}
    need = max(1, (a.votes + 1) // 2)

    def work(q):
        # Circuit breaker. If the API stops answering, every question turns into an error
        # record and the run would otherwise burn through the rest of the corpus producing
        # nothing. Stop early and leave the remaining questions unprocessed so a resume can
        # pick them up once the endpoint is healthy.
        if counter.get("abort"):
            return
        rec = {"id": q["id"], "arity": q.get("arity"), "n_options": q.get("n_options")}
        try:
            opts, ans = q["options"], sorted(q["answer"])
            gold_texts = [opts[k] for k in ans]
            n_wrong = len(opts) - len(ans)

            # ---- stage 1: is it already sound? ----
            hits, guesses = blind(api, q["question"], opts, ans, a.votes)
            if hits < need:
                rec.update({"stage": "original", "status": "passed",
                            "options": opts, "answer": ans, "error": None})
                return finish(rec)

            ev = caps.get((q.get("video"), tuple(q.get("span_sec") or [])))
            if ev is None:
                raise RuntimeError("no evidence")
            clen = [len(t) for t in gold_texts]
            lo, hi = min(clen) * 0.75, max(clen) * 1.05
            gold_norm = {t.strip().lower() for t in gold_texts}
            evs = json.dumps(ev, ensure_ascii=False)

            # ---- stage 2: v1 ----
            w = json_call(api, REPAIR_PROMPT.format(
                evidence=evs, question=q["question"], correct="\n".join(gold_texts),
                n=n_wrong, lo=int(lo), hi=int(hi)),
                lambda d: validate_repair(d, n_wrong, lo, hi, gold_norm))
            o1, a1 = shuffle_letters(gold_texts + w, set(range(len(gold_texts))), q["question"])
            h1, g1 = blind(api, q["question"], o1, a1, a.votes)
            if h1 < need:
                rec.update({"stage": "v1", "status": "passed",
                            "options": o1, "answer": a1, "error": None})
                return finish(rec)

            # ---- stage 3: v2, adversarial ----
            prev_set, prev_guess = w, ", ".join(o1.get(k, "?") for k in (g1[0] or [])) or \
                                      ", ".join(gold_texts)
            for rnd in range(1, a.v2_rounds + 1):
                fb = FEEDBACK_TMPL.format(guess=prev_guess,
                                          prev="\n".join(f"  - {t}" for t in prev_set))
                w2 = json_call(api, REPAIR_V2.format(
                    evidence=evs, question=q["question"], correct="\n".join(gold_texts),
                    n=n_wrong, lo=int(lo), hi=int(hi), feedback=fb),
                    lambda d: validate_repair(d, n_wrong, lo, hi, gold_norm))
                verdicts, ncon = verify_contradicted(api, ev, w2)
                if ncon < a.require_contradicted * n_wrong and rnd < a.v2_rounds:
                    prev_set = w2
                    continue
                o2, a2 = shuffle_letters(gold_texts + w2, set(range(len(gold_texts))), q["question"])
                h2, g2 = blind(api, q["question"], o2, a2, a.votes)
                if h2 < need:
                    rec.update({"stage": f"v2_r{rnd}", "status": "passed",
                                "options": o2, "answer": a2, "contradicted": ncon,
                                "error": None})
                    return finish(rec)
                prev_set = w2
                prev_guess = ", ".join(o2.get(k, "?") for k in (g2[0] or [])) or prev_guess

            # ---- stage 4: revert, do not ship an unproven rewrite ----
            rec.update({"stage": "exhausted", "status": "still_guessable",
                        "options": opts, "answer": ans, "error": None})
            return finish(rec)
        except Exception as exc:
            rec.update({"error": f"{type(exc).__name__}: {exc}"})
            return finish(rec)

    ABORT_AFTER = 25

    def finish(rec):
        with lock:
            if rec.get("error"):
                counter["consec"] = counter.get("consec", 0) + 1
                if counter["consec"] >= ABORT_AFTER and not counter.get("abort"):
                    counter["abort"] = True
                    print(f"  ABORT: {ABORT_AFTER} consecutive errors — endpoint looks "
                          f"unhealthy; stopping so the rest can be resumed later. "
                          f"last: {rec['error'][:120]}", flush=True)
            else:
                counter["consec"] = 0
            done[rec["id"]] = rec
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            counter["n"] += 1
            i = counter["n"]
            if i % 200 == 0:
                ok = [r for r in done.values() if not r.get("error")]
                p = sum(1 for r in ok if r["status"] == "passed")
                el = (time.time() - t0) / 60
                rate = i / el if el else 0
                print(f"  [{i}/{len(todo)}] passed {p}/{len(ok)} = {100*p/max(1,len(ok)):.1f}%  "
                      f"{el:.0f}m  {rate:.1f}/min  ETA {(len(todo)-i)/max(rate,.01)/60:.1f}h",
                      flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(work, todo))
    fh.close()

    ok = [r for r in done.values() if not r.get("error")]
    n = len(ok)
    p = sum(1 for r in ok if r["status"] == "passed")
    lo_, hi_ = wilson(p, n) if n else (0, 0)
    summary = {
        "processed_this_run": n, "errors": len(done) - n,
        "passed": p, "pass_rate_pct": round(100 * p / n, 1) if n else None,
        "pass_ci95": [round(100 * lo_, 1), round(100 * hi_, 1)],
        "by_stage": dict(collections.Counter(r["stage"] for r in ok)),
        "by_arity_passed": dict(collections.Counter(
            r["arity"] for r in ok if r["status"] == "passed")),
        "settled_by_earlier_passes": len(settled),
    }
    Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
