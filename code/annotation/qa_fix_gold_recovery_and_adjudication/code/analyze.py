#!/usr/bin/env python3
"""Cross-tabulate the three independent views of every question:
   recovered gold  x  human annotator  x  Gemini video recheck  x  blind gate."""
import json, os, collections, sys

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
TRUSTED_EXCLUDE = {"Junwei Zheng"}   # option overlap 43.6% vs 37.5% random -> no signal


def load():
    idx = {json.loads(l)["review_id"]: json.loads(l) for l in open(f"{QA}/master_index.jsonl")}
    vid = {}
    p = f"{QA}/video_review.jsonl"
    if os.path.exists(p):
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not r.get("error"):
                vid.setdefault(r["review_id"], {})[r["vote"]] = r
    blind = {}
    p = f"{QA}/blind_gate.jsonl"
    if os.path.exists(p):
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not r.get("error"):
                blind[r["review_id"]] = r
    return idx, vid, blind


def pct(a, b):
    return f"{100*a/b:.1f}%" if b else "n/a"


def main():
    idx, vid, blind = load()
    print(f"corpus={len(idx)}  video_reviewed={len(vid)}  blind_gated={len(blind)}\n")

    # --- 1. Gemini video review vs recovered gold -------------------------------
    n = ex = ov = 0
    gold_opt_ruled = gold_opt_tot = 0
    verd = collections.Counter()
    for rid, votes in vid.items():
        r = votes.get(0)
        if not r:
            continue
        g = set(idx[rid]["gold"])
        verd[r["verdict"]] += 1
        if not g:
            continue
        n += 1
        ex += set(r["final_answer"]) == g
        ov += len(set(r["final_answer"]) & g) / max(1, len(g))
        for l in g:
            gold_opt_tot += 1
            gold_opt_ruled += r["per_option"].get(l) == "ruled_out"
    print("== Gemini video recheck vs recovered gold ==")
    print(f"  scoreable {n}  exact {pct(ex,n)}  mean option overlap {pct(ov,n)}")
    print(f"  gold options the footage RULES OUT: {gold_opt_ruled}/{gold_opt_tot} = {pct(gold_opt_ruled,gold_opt_tot)}")
    print(f"  gemini verdict: {dict(verd)}")

    # --- 2. Gemini vs human (the calibration that matters) ----------------------
    n = ex = ov = 0
    agree_verdict = collections.Counter()
    for rid, votes in vid.items():
        r = votes.get(0)
        hs = [h for h in idx[rid]["human"] if h["annotator"] not in TRUSTED_EXCLUDE]
        if not r or not hs:
            continue
        h = hs[0]
        n += 1
        ex += set(r["final_answer"]) == set(h["final_answer"])
        ov += len(set(r["final_answer"]) & set(h["final_answer"])) / max(1, len(h["final_answer"]) or 1)
        agree_verdict[(h["verdict"], r["verdict"])] += 1
    print("\n== Gemini video recheck vs human annotator (independent, both saw footage) ==")
    print(f"  paired {n}  exact answer match {pct(ex,n)}  overlap {pct(ov,n)}")
    for k, v in sorted(agree_verdict.items(), key=lambda x: -x[1])[:9]:
        print(f"    human={k[0]:10s} gemini={k[1]:10s} {v}")

    # --- 3. three-way: does gold survive when BOTH independent views reject it? --
    both_reject = gold_ok = 0
    rekey = 0
    for rid, votes in vid.items():
        r = votes.get(0)
        hs = [h for h in idx[rid]["human"] if h["annotator"] not in TRUSTED_EXCLUDE]
        g = set(idx[rid]["gold"])
        if not r or not hs or not g:
            continue
        h = hs[0]
        hr = {l for l, v in (h["per_option"] or {}).items() if v == "ruled_out"}
        gr = {l for l, v in r["per_option"].items() if v == "ruled_out"}
        if g & hr and g & gr:
            both_reject += 1
        if set(r["final_answer"]) == set(h["final_answer"]) == g:
            gold_ok += 1
        elif set(r["final_answer"]) == set(h["final_answer"]) != g and len(h["final_answer"]) == idx[rid]["n_select"]:
            rekey += 1
    print("\n== three-way ==")
    print(f"  gold confirmed by BOTH human and gemini: {gold_ok}")
    print(f"  human and gemini agree on a DIFFERENT answer than gold (re-key candidates): {rekey}")
    print(f"  a gold option ruled out by BOTH: {both_reject}")

    # --- 4. blind gate ----------------------------------------------------------
    if blind:
        scored = [b for b in blind.values() if "guessable" in b]
        g = sum(b["guessable"] for b in scored)
        print(f"\n== blind gate (no video) ==\n  scored {len(scored)}  guessable(2of3) {pct(g,len(scored))}")
        by_n = collections.defaultdict(lambda: [0, 0])
        for b in scored:
            k = f"{b['n_options']}opt/pick{b['n_select']}"
            by_n[k][0] += b["guessable"]; by_n[k][1] += 1
        for k in sorted(by_n):
            a, t = by_n[k]
            print(f"    {k:14s} {a}/{t} = {pct(a,t)}")


if __name__ == "__main__":
    main()
