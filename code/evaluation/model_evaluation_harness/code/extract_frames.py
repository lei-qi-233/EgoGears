#!/usr/bin/env python3
"""One visual budget for every model, spent per QUESTION rather than per video.

RunningBench questions carry 1 to 10 clips. Fixing frames-per-video would hand a
ten-clip question ten times the evidence of a one-clip question, and would make the
per-model numbers incomparable the moment two models sample differently. So the budget
is TOTAL_FRAMES per question, split across that question's clips in proportion to
duration, with a floor so no clip vanishes.

Frames are decoded once, here, and every model is then shown the same JPEGs.
"""
import argparse, json, os, re, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
OUT = "/mnt/data/cvhci_video_understanding/eval/frames"
TOTAL_FRAMES = 64
MIN_PER_CLIP = 4
MAX_SIDE = 480


def duration(path):
    out = subprocess.run(["ffmpeg", "-i", path], capture_output=True, timeout=120).stderr.decode("utf-8", "replace")
    m = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", out)
    return float(m.group(1)) * 3600 + float(m.group(2)) * 60 + float(m.group(3)) if m else 60.0


def allocate(durs, total, floor):
    """Largest-remainder apportionment, then lift everyone to the floor."""
    n = len(durs)
    if n * floor >= total:
        return [floor] * n
    s = sum(durs) or 1.0
    raw = [d / s * total for d in durs]
    base = [max(floor, int(x)) for x in raw]
    while sum(base) > total:
        i = max(range(n), key=lambda i: (base[i] - floor, raw[i]))
        if base[i] <= floor:
            break
        base[i] -= 1
    rem = total - sum(base)
    order = sorted(range(n), key=lambda i: -(raw[i] - int(raw[i])))
    for i in range(rem):
        base[order[i % n]] += 1
    return base


def one(rec):
    rid = rec["review_id"]
    d = f"{OUT}/{rid}"
    marker = f"{d}/.ok.json"
    if os.path.exists(marker):
        try:
            return json.load(open(marker))
        except Exception:
            pass
    os.makedirs(d, exist_ok=True)
    clips = rec["clips"]
    try:
        durs = [duration(c["file"]) for c in clips]
        quota = allocate(durs, TOTAL_FRAMES, MIN_PER_CLIP)
        out = {"review_id": rid, "clips": []}
        for i, (c, dur, k) in enumerate(zip(clips, durs, quota)):
            # Name by clip INDEX, never by label. cross_video shows one recording as
            # several excerpts that all carry that recording's letter, so 40 questions
            # have repeated labels -- keying files by label made two clips write the same
            # filenames, overwriting each other and double-counting on the way back in.
            fps = k / max(dur, 0.1)
            pre = f"c{i:02d}"
            pat = f"{d}/{pre}_%03d.jpg"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-threads", "1",
                            "-i", c["file"],
                            "-vf", f"fps={fps:.6f},scale='min({MAX_SIDE},iw)':-2",
                            "-frames:v", str(k), "-q:v", "3", pat],
                           check=True, capture_output=True, timeout=900)
            files = sorted(f"{d}/{f}" for f in os.listdir(d)
                           if f.startswith(pre + "_") and f.endswith(".jpg"))
            out["clips"].append({"label": c["label"], "frames": files, "n": len(files),
                                 "duration_s": round(dur, 2), "quota": k})
        out["n_frames"] = sum(x["n"] for x in out["clips"])
        out["bytes"] = sum(os.path.getsize(f) for x in out["clips"] for f in x["frames"])
        json.dump(out, open(marker, "w"))
        return out
    except Exception as exc:
        return {"review_id": rid, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--corpus", default=f"{QA}/runningbench_v2_kept.jsonl")
    a = ap.parse_args()
    prep = {}
    for l in open(f"{QA}/clipprep.jsonl"):
        if l.strip():
            r = json.loads(l)
            prep[r["review_id"]] = r
    recs = []
    for l in open(a.corpus):
        q = json.loads(l)
        p = prep.get(q["review_id"])
        if p and not p.get("error"):
            recs.append({"review_id": q["review_id"], "clips": p["clips"]})
    os.makedirs(OUT, exist_ok=True)
    print(f"questions={len(recs)} budget={TOTAL_FRAMES} frames/question floor={MIN_PER_CLIP} max_side={MAX_SIDE}", flush=True)
    done = errs = 0
    with open(f"{OUT}/../frames_index.jsonl", "a") as fh, ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(one, r) for r in recs]):
            r = f.result()
            fh.write(json.dumps(r) + "\n"); fh.flush()
            done += 1; errs += bool(r.get("error"))
            if done % 100 == 0:
                print(f"{done}/{len(recs)} errors={errs}", flush=True)
    print(f"DONE {done} errors={errs}", flush=True)


if __name__ == "__main__":
    main()
