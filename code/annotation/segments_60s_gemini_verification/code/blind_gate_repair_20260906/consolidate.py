#!/usr/bin/env python3
"""Fold every measurement pass into one corpus file, keeping the best known version of each question.

Four passes have touched this corpus, each with its own file and its own idea of what a
question currently looks like. This resolves them into one row per question, choosing the
version that actually survived a blind test, and records which pass produced it so any row
can be traced back.

Precedence, best first:
  0. the unified pipeline's verdict           -> whatever stage cleared it (or the original)
  1. v2 repair that defeated the blind model  -> v2 options
  2. v1 repair that defeated it               -> v1 options
  3. never guessable to begin with            -> ORIGINAL options, untouched
  4. nothing worked                           -> original options, flagged unusable

The pipeline runs all three stages itself and already reverts what it could not fix, so its
row supersedes the older per-stage files wherever it exists. The earlier files still carry
the 969 questions settled before the pipeline existed.

Rule 3 matters: a question the model could not guess in the first place must keep its
original wording. v1 rewrote unconditionally and made 13 of 432 single-choice questions
worse that way, so anything that was already sound is reverted here rather than kept.
"""
import collections, json, sys
from pathlib import Path

D = Path(__file__).parent


def load(name):
    p = D / name
    if not p.exists():
        return {}
    out = {}
    for l in p.open():
        try:
            r = json.loads(l)
        except Exception:
            continue
        if not r.get("error") and r.get("id"):
            out[r["id"]] = r
    return out


def main():
    qdir = sys.argv[1]
    qs = {q["id"]: q for q in (json.loads(l) for l in open(f"{qdir}/questions/segments_60s.jsonl"))}
    gate = load("segments_60s_gate.progress.jsonl")
    v1 = load("repair_measure.progress.jsonl")
    v2 = load("repair_v2.progress.jsonl")
    singles = load("singles_repaired.progress.jsonl")
    pipe = load("pipeline.progress.jsonl")

    rows, stats = [], collections.Counter()
    covered = set(gate) | set(v1) | set(v2) | set(singles) | set(pipe)
    for qid in sorted(covered):
        q = qs.get(qid)
        if not q:
            continue
        opts, ans = q["options"], sorted(q["answer"])
        source, status = None, None

        w2, w1, ws, g = v2.get(qid), v1.get(qid), singles.get(qid), gate.get(qid)
        wp = pipe.get(qid)

        if wp:
            # The pipeline already measured, escalated and reverted; take its verdict whole.
            opts, ans = wp["options"], wp["answer"]
            status = "passed" if wp["status"] == "passed" else "still_guessable"
            source = {"original": "original_never_guessable", "v1": "v1_repair",
                      "exhausted": "original_unsalvaged"}.get(
                          wp["stage"], "v2_repair" if wp["stage"].startswith("v2") else wp["stage"])
        elif w2 and not w2["v2_guessable"]:
            opts, ans, source, status = w2["options"], w2["answer"], "v2_repair", "passed"
        elif w1 and w1.get("control_guessable") and not w1.get("treat_guessable"):
            opts, ans, source, status = (w1["new_options"], w1["new_answer"],
                                         "v1_repair", "passed")
        elif ws and ws.get("repaired") and not ws.get("blind_after"):
            opts, ans, source, status = ws["options"], ws["answer"], "v1_repair", "passed"
        else:
            # A question can be measured by more than one pass, and the passes disagree on
            # 15.8% of the ones measured twice -- 1 vote at temperature 0.2 is a noisy
            # verdict. Requiring EVERY available measurement to clear it, rather than any
            # single one, keeps the noise from being read as a pass: the lenient reading
            # admits 110 more questions here, none of them on new evidence.
            seen = []
            if w1:
                seen.append(not w1.get("control_guessable"))
            if ws:
                seen.append(not ws.get("blind_before"))
            if g:
                seen.append(not g.get("blind_guessable"))
            if seen and all(seen):
                source, status = "original_never_guessable", "passed"  # keep as written
            elif seen and any(seen):
                source, status = "original_disputed", "still_guessable"
            else:
                source, status = "original_unsalvaged", "still_guessable"

        stats[f"{status}/{source}"] += 1
        stats[status] += 1
        rows.append({
            "id": qid, "video": q.get("video"), "span_sec": q.get("span_sec"),
            "question": q["question"], "options": opts, "answer": ans,
            "arity": q.get("arity"), "n_options": len(opts),
            "provenance": q.get("provenance"), "unit": q.get("unit"),
            "source": source, "screening_status": status,
        })

    out = D / "runningbench_segments_curated.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    passed = [r for r in rows if r["screening_status"] == "passed"]
    by_arity = collections.Counter(r["arity"] for r in passed)
    summary = {
        "questions_resolved": len(rows),
        "passed_blind_gate": len(passed),
        "still_guessable": len(rows) - len(passed),
        "pass_rate_pct": round(100 * len(passed) / len(rows), 1) if rows else None,
        "passed_by_arity": dict(by_arity),
        "by_source": {k: v for k, v in sorted(stats.items()) if "/" in k},
        "corpus_total_segments_60s": len(qs),
        "coverage_pct": round(100 * len(rows) / len(qs), 1),
    }
    (D / "curated_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
