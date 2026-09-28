#!/usr/bin/env python3
"""Do late-round repairs hold up? An independent re-check, split by how many tries it took.

The worry is mechanical: a rewrite is sampled, then judged by a fixed gate. Re-sampling the
same question until something passes is multiple testing -- the more draws, the larger the
share of passes that are luck rather than merit. If that is happening, repairs that needed
four rounds should survive an independent re-check less often than ones that passed first
try.

The auditor is gemini-3.5-flash: it wrote none of these options and verified none of them
(it was only ever the text-only blind gate), so it is the one model with no stake here.
It is shown the repaired options and never the key.
"""
import argparse, json, os, random, sys, threading, collections
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floodgate import Floodgate, RateLimiter
from video_review import review as video_review_call

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
AUDITOR = "gemini-3.5-flash"


def load_repairs():
    out = {}
    for f in ("repairs.jsonl", "repairs_crossmodel.jsonl", "repairs_deep.jsonl"):
        p = f"{QA}/{f}"
        if not os.path.exists(p):
            continue
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("repaired"):
                r["_src"] = f
                out[r["review_id"]] = r          # later file wins, same as the pipeline
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=82)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rps", type=float, default=0.45)
    ap.add_argument("--out", default=f"{QA}/audit_rounds.jsonl")
    a = ap.parse_args()

    idx = {json.loads(l)["review_id"]: json.loads(l) for l in open(f"{QA}/master_index.jsonl")}
    prep = {}
    for l in open(f"{QA}/clipprep.jsonl"):
        if l.strip():
            r = json.loads(l)
            prep[r["review_id"]] = r
    rep = load_repairs()
    early = [r for r in rep.values() if r.get("rounds_used") == 1 and r["review_id"] in prep]
    late = [r for r in rep.values() if (r.get("rounds_used") or 0) >= 3 and r["review_id"] in prep]
    random.seed(20260917)
    early = random.sample(early, min(a.per_group, len(early)))
    late = random.sample(late, min(a.per_group, len(late)))
    todo = [("early", r) for r in early] + [("late", r) for r in late]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(json.loads(l)["review_id"])
            except Exception:
                pass
    todo = [t for t in todo if t[1]["review_id"] not in done]
    print(f"auditor={AUDITOR} early={len(early)} late={len(late)} todo={len(todo)}", flush=True)

    api = Floodgate(limiter=RateLimiter(a.rps))
    lock = threading.Lock()

    def one(group, r):
        rid = r["review_id"]
        # the question as delivered: repaired options, repaired key held back
        probe = dict(idx[rid], options=r["options"], gold=r["answer"])
        vr = video_review_call(api, probe, prep[rid], vote=11, model=AUDITOR)
        if vr.get("error"):
            return {"review_id": rid, "group": group, "error": vr["error"]}
        return {"review_id": rid, "group": group, "rounds_used": r.get("rounds_used"),
                "src": r["_src"], "writer": r.get("writer_model"), "verifier": r.get("verify_model"),
                "key": sorted(r["answer"]), "auditor_answer": sorted(vr["final_answer"]),
                "auditor_verdict": vr["verdict"],
                "match": sorted(vr["final_answer"]) == sorted(r["answer"])}

    n = 0
    with open(a.out, "a") as fh, ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(one, g, r) for g, r in todo]):
            res = f.result()
            with lock:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush()
                n += 1
                if n % 20 == 0:
                    print(f"{n}/{len(todo)}", flush=True)
    print(f"DONE {n}", flush=True)


if __name__ == "__main__":
    main()
