#!/usr/bin/env python3
"""Blind-guess gate: answer each question with NO video at all.

A question a model answers right without ever seeing the footage is not a video
understanding question. Prompt and the 2-of-3 rule are kept verbatim from the
project's own gate so the number stays comparable with the historical 48.6%.
"""
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from floodgate import Floodgate, RateLimiter, STRUCTURE_MODEL, clean_json

QA = "/mnt/data/cvhci_video_understanding/qa_fix"

BLIND_PROMPT = """Answer this multiple-choice question WITHOUT any video — you have none. Return only
JSON: {{"answer": ["..."]}}

QUESTION: {question}
OPTIONS: {options}
{arity}"""


def one_vote(api, rec, vote, model=STRUCTURE_MODEL):
    letters = sorted(rec["options"])
    opts = "\n".join(f"{l}. {rec['options'][l]}" for l in letters)
    n = rec["n_select"]
    arity = f"Select exactly {n} option{'s' if n > 1 else ''}."
    parts = [{"text": BLIND_PROMPT.format(question=rec["question"], options=opts, arity=arity)}]
    raw = api.generate(model, parts, max_tokens=2048, timeout=180, attempts=4,
                       temperature=0.2)
    d = clean_json(raw) or {}
    ans = d.get("answer") or []
    if isinstance(ans, str):
        ans = [ans]
    return sorted({a.strip()[0].upper() for a in ans if isinstance(a, str) and a.strip()} & set(letters))


def run(api, rec, votes):
    rid = rec["review_id"]
    out = {"id": rid, "review_id": rid, "bundle": rec["bundle"], "n_select": rec["n_select"],
           "n_options": len(rec["options"]), "question_type": rec["question_type"],
           "gold": rec["gold"], "votes": []}
    try:
        for v in range(votes):
            out["votes"].append(one_vote(api, rec, v))
    except Exception as exc:
        return {"id": rid, "review_id": rid, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    gold = set(rec["gold"])
    if gold:
        hits = sum(1 for v in out["votes"] if set(v) == gold)
        out["hits"] = hits
        out["guessable"] = hits >= 2 if votes >= 3 else hits >= 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rps", type=float, default=2.0)
    ap.add_argument("--out", default=f"{QA}/blind_gate.jsonl")
    ap.add_argument("--only", default="", help="file of review_ids, one per line")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(f"{QA}/master_index.jsonl")]
    if a.only:
        keep = {l.strip() for l in open(a.only) if l.strip()}
        recs = [r for r in recs if r["review_id"] in keep]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not r.get("error") and r.get("id"):
                done.add(r["id"])
    todo = [r for r in recs if r["review_id"] not in done]
    print(f"todo={len(todo)} votes={a.votes}", flush=True)
    api = Floodgate(limiter=RateLimiter(a.rps))
    lock = threading.Lock()
    n = 0
    with open(a.out, "a") as fh, ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run, api, r, a.votes) for r in todo]):
            r = f.result()
            with lock:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
                n += 1
                if n % 100 == 0:
                    print(f"{n}/{len(todo)}", flush=True)
    print(f"DONE {n}", flush=True)


if __name__ == "__main__":
    main()
