#!/usr/bin/env python3
"""Measure how guessable the 60-second segment questions actually are.

The segment corpus has never faced a blind model. Its published "17.6% blind
baseline" is only the answer-letter distribution -- it shows the shuffle removed
positional bias, not that the questions need the video. The full-video corpus,
which does run this test, found 6-option single-choice questions hit 41% blind:
2.5x the 16.7% random baseline, and the leak is semantic, not positional.

So this runs the identical procedure the full-video gates use -- same prompt,
same three-vote rule, guessable at >=2/3 -- over a stratified sample, and reports
a Wilson interval.

Usage:  blind_test_segments.py [--n 800] [--out report.json]
"""
import argparse
import collections
import glob
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

SEG_ROOT = "/mnt/data/data_anno/runningbench_qa_expansion/annotations_60s"
BASE = Path("/mnt/task_runtime/bolt/gdrive_relay")
QUOTA_PER_SECOND = 0.45
SEED = 20260830


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", default="/mnt/data/data_anno/runningbench_qa_expansion/blind_test.json")
    a = ap.parse_args()

    fv = load("fv", BASE / "build_fullvideo_annotations.py")
    gen = fv.load_generator()

    pool = []
    for path in sorted(glob.glob(f"{SEG_ROOT}/**/*.json", recursive=True)):
        try:
            record = json.load(open(path))
        except Exception:
            continue
        for item in record.get("qa_pairs") or []:
            opts, ans = item.get("options"), item.get("answer")
            if not isinstance(opts, dict) or not opts or not isinstance(ans, list) or not ans:
                continue
            pool.append({"question": item["question"], "options": opts, "answer": sorted(ans),
                         "type": item.get("type"), "n_opt": len(opts), "n_ans": len(ans),
                         "file": path})
    print(f"pool: {len(pool)} objective questions", flush=True)

    # Stratify by (option count, single/multi) so the sample mirrors the corpus.
    strata = collections.defaultdict(list)
    for q in pool:
        strata[(q["n_opt"], q["n_ans"] > 1)].append(q)
    rng = random.Random(SEED)
    sample = []
    for key, group in sorted(strata.items()):
        take = max(1, round(a.n * len(group) / len(pool)))
        rng.shuffle(group)
        sample += group[:take]
    rng.shuffle(sample)
    print(f"sample: {len(sample)} over {len(strata)} strata", flush=True)

    api = gen.Floodgate("")
    api.session = fv.PacedSession(api.session, fv.RateLimiter(QUOTA_PER_SECOND))

    results = []
    for i, q in enumerate(sample, 1):
        arity = ("Exactly one option is correct." if q["n_ans"] == 1
                 else "More than one option is correct.")
        guesses, hits = [], 0
        for attempt in range(3):
            try:
                raw = gen.generate_valid_json(
                    api,
                    fv.BLIND_PROMPT.format(question=q["question"],
                                           options=json.dumps(q["options"], ensure_ascii=False),
                                           arity=f"{arity} (attempt {attempt + 1})"),
                    fv.validate_blind, 4096)
                g = sorted(raw["answer"])
            except Exception as exc:
                g = [f"ERROR:{type(exc).__name__}"]
            guesses.append(g)
            hits += g == q["answer"]
        results.append({**{k: q[k] for k in ("question", "type", "n_opt", "n_ans", "answer")},
                        "guesses": guesses, "hits_of_3": hits, "guessable": hits >= 2})
        if i % 25 == 0 or i == len(sample):
            g = sum(1 for r in results if r["guessable"])
            print(f"  {i}/{len(sample)}  guessable {g} ({100 * g / i:.1f}%)", flush=True)
        json.dump({"n": len(results), "results": results}, open(a.out, "w"), ensure_ascii=False)

    # ---- report -----------------------------------------------------------
    def rate(rows):
        if not rows:
            return None
        k = sum(1 for r in rows if r["guessable"])
        lo, hi = wilson(k, len(rows))
        return k, len(rows), 100 * k / len(rows), 100 * lo, 100 * hi

    print("\n=== blind-guess rate (>=2 of 3 votes match) ===")
    k, n, p, lo, hi = rate(results)
    print(f"overall           {k}/{n} = {p:.1f}%   95% CI [{lo:.1f}, {hi:.1f}]")
    for key, label in (((6, False), "6-option single"), ((8, True), "8-option multi")):
        rows = [r for r in results if (r["n_opt"], r["n_ans"] > 1) == key]
        if rows:
            k, n, p, lo, hi = rate(rows)
            rand = 100 / math.comb(key[0], rows[0]["n_ans"]) if key[1] else 100 / key[0]
            print(f"{label:18s}{k}/{n} = {p:.1f}%   95% CI [{lo:.1f}, {hi:.1f}]   random ~{rand:.1f}%")
    print("\nby question type:")
    for t in sorted({r["type"] for r in results}):
        rows = [r for r in results if r["type"] == t]
        k, n, p, lo, hi = rate(rows)
        print(f"  {str(t):30s} {k:3d}/{n:3d} = {p:5.1f}%   [{lo:.1f}, {hi:.1f}]")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
