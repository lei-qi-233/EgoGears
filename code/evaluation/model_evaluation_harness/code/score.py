#!/usr/bin/env python3
"""Score the runs. Two metrics and, crucially, the right random baseline.

RunningBench mixes 3..8 options with 1..5 correct, so the chance line is NOT 1/n and NOT
one number for the whole set: it is C(n_options, n_select) per question, averaged over the
questions each model actually answered. Reporting a single total against a single baseline
lets a model that simply prefers single-answer questions look better than it is.
"""
import argparse, collections, json, math, os, sys
from math import comb

EVAL = "/mnt/data/cvhci_video_understanding/eval"


def load(tag):
    rows = []
    for l in open(f"{EVAL}/results/{tag}.jsonl"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if not r.get("error"):
            rows.append(r)
    ded = {r["review_id"]: r for r in rows}      # a resumed run can repeat a question
    return list(ded.values())


def baselines(rows):
    """Exact-match chance is averaged over questions; overlap chance is averaged over
    answer SLOTS, which is a different weighting and easy to get wrong."""
    ex = sum(1.0 / comb(r["n_options"], r["n_select"]) for r in rows) / len(rows)
    num = sum(r["n_select"] ** 2 / r["n_options"] for r in rows)
    den = sum(r["n_select"] for r in rows)
    return ex, num / den


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--by", default="unit", choices=["unit", "question_type", "n_clips", "spec"])
    a = ap.parse_args()
    tags = a.tags or sorted(x[:-6] for x in os.listdir(f"{EVAL}/results") if x.endswith(".jsonl"))
    print(f"{'model':32s} {'n':>5s} {'exact':>7s} {'95% CI':>14s} {'chance':>7s} {'overlap':>8s} {'chance':>7s} {'多选够数':>8s}")
    allrows = {}
    for t in tags:
        rows = load(t)
        if not rows:
            continue
        allrows[t] = rows
        n = len(rows); ex = sum(r["exact"] for r in rows)
        ov = sum(r["overlap"] for r in rows) / n
        bex, bov = baselines(rows)
        lo, hi = wilson(ex, n)
        wellformed = sum(1 for r in rows if r["n_pred"] == r["n_select"]) / n
        print(f"{t:32s} {n:5d} {100*ex/n:6.2f}% {100*lo:5.1f}-{100*hi:5.1f}% {100*bex:6.2f}% "
              f"{100*ov:7.1f}% {100*bov:6.1f}% {100*wellformed:7.1f}%")
    if not allrows:
        return
    key = {"unit": "unit", "question_type": "question_type", "n_clips": "n_clips"}.get(a.by)
    print(f"\n--- 按 {a.by} 拆分（exact）---")
    cats = sorted({(r[key] if key != "n_clips" else min(r["n_clips"], 5)) for rs in allrows.values() for r in rs},
                  key=str)
    hdr = "".join(f"{str(c)[:16]:>17s}" for c in cats)
    print(f"{'model':32s}{hdr}")
    for t, rows in allrows.items():
        g = collections.defaultdict(list)
        for r in rows:
            g[r[key] if key != "n_clips" else min(r["n_clips"], 5)].append(r["exact"])
        cells = "".join(f"{(100*sum(g[c])/len(g[c]) if g[c] else float('nan')):16.1f}%" for c in cats)
        print(f"{t:32s}{cells}")
    print("\n注: n_clips 一列的 5 表示 >=5 个片段。chance 为按题目规格 C(n_options, n_select) 实算。")


if __name__ == "__main__":
    main()
