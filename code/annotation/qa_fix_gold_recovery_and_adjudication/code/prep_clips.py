#!/usr/bin/env python3
"""Make every question's clip set inline-able (<=13 MB raw -> <=17.3 MB base64 < 20 MB cap).

Two facts decide the design:
  * the bundle clips are already 480p h264, and Gemini accepts them untouched
    (10-bit High 10 included) -- so 863 of the 1,526 questions need no work at all;
  * Gemini samples video at roughly 1 fps, so for the 663 over-budget questions the
    cheapest lever by far is frame rate, not resolution. 480p is kept (360p makes the
    model call readable signage "blurry" and rule out true options); fps goes to 5.
"""
import json, os, re, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
CACHE = f"{QA}/clipcache"
BUDGET = 13 * 1024 * 1024
# The durations table lived on the machine that built the bundles. It is an optimisation,
# not a dependency: duration_of falls back to the clip filename, then to probing the file.
try:
    DURATIONS = json.load(open("/mnt/task_runtime/runningbench/clip_durations.json"))
except Exception:
    DURATIONS = {}


def duration_of(bundle, relpath, src):
    d = DURATIONS.get(f"{bundle}|{relpath}")
    if d:
        return float(d)
    m = re.search(r"_(\d+)_(\d+)\.mp4$", os.path.basename(src))   # LABEL_start_end.mp4
    if m:
        return max(1.0, float(m.group(2)) - float(m.group(1)))
    # anonymous_multi_clip clips are plain CLIP_A.mp4 with no span in the name. Probing
    # beats defaulting: a wrong duration only mis-sizes the bitrate, but by a lot.
    # ffprobe is not installed here, so read it off ffmpeg's own stderr.
    try:
        out = subprocess.run(["ffmpeg", "-i", src], capture_output=True, timeout=120).stderr.decode("utf-8", "replace")
        m = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", out)
        if m:
            h, mi, sec = float(m.group(1)), float(m.group(2)), float(m.group(3))
            return max(1.0, h * 3600 + mi * 60 + sec)
    except Exception:
        pass
    return 60.0


def encode(src, dst, kbps, fps=5):
    """One pass at a computed bitrate. A crf ladder re-encodes the same clip up to six
    times; on a box already at load 400 that is the whole cost of the job."""
    # -threads must be given for BOTH decode (before -i) and encode (after -i), and the
    # filter graph needs its own cap. ffmpeg's default is one thread per core: on a box
    # already at load 400 that turns a 1.7 s encode into 19 minutes, 99% of it in sys.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1",
                    "-i", src,
                    "-vf", f"scale=-2:480,fps={fps}", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-preset", "veryfast", "-threads", "1",
                    "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps*1.3)}k",
                    "-bufsize", f"{int(kbps*2)}k", "-an",
                    "-movflags", "+faststart", dst], check=True, capture_output=True, timeout=1800)
    return os.path.getsize(dst)


def prep(rec):
    rid = rec["review_id"]
    marker = f"{CACHE}/{rid}/.ok.json"
    if os.path.exists(marker):
        try:
            return json.load(open(marker))
        except Exception:
            pass
    clips = [dict(label=c["label"], src=os.path.join(rec["media_root"], c["path"])) for c in rec["clips"]]
    for c in clips:
        c["src_bytes"] = os.path.getsize(c["src"])
    total = sum(c["src_bytes"] for c in clips)
    if total <= BUDGET:
        out = {"review_id": rid, "mode": "passthrough", "src_bytes": total, "total_bytes": total,
               "clips": [{"label": c["label"], "file": c["src"], "bytes": c["src_bytes"]} for c in clips]}
        os.makedirs(f"{CACHE}/{rid}", exist_ok=True)
        json.dump(out, open(marker, "w"))
        return out
    os.makedirs(f"{CACHE}/{rid}", exist_ok=True)
    share = BUDGET / len(clips)
    res = {"review_id": rid, "mode": "transcode", "src_bytes": total, "clips": []}
    try:
        for c in clips:
            dst = f"{CACHE}/{rid}/{c['label']}.mp4"
            if c["src_bytes"] <= share:                      # already under its share
                res["clips"].append({"label": c["label"], "file": c["src"], "bytes": c["src_bytes"], "crf": None})
                continue
            dur = duration_of(rec["bundle"], next(x["path"] for x in rec["clips"] if x["label"] == c["label"]), c["src"])
            kbps = max(120, int(share * 0.92 * 8 / 1000 / dur))
            size = encode(c["src"], dst, kbps)
            if size > share:                                  # bitrate overshoot, one correction
                kbps = max(90, int(kbps * share / size * 0.9))
                size = encode(c["src"], dst, kbps)
            res["clips"].append({"label": c["label"], "file": dst, "bytes": size, "kbps": kbps})
        res["total_bytes"] = sum(c["bytes"] for c in res["clips"])
        json.dump(res, open(marker, "w"))
        return res
    except Exception as exc:
        return {"review_id": rid, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


if __name__ == "__main__":
    recs = [json.loads(l) for l in open(f"{QA}/master_index.jsonl")]
    os.makedirs(CACHE, exist_ok=True)
    # cheap ones first so the video pass can start immediately
    recs.sort(key=lambda r: sum(os.path.getsize(os.path.join(r["media_root"], c["path"])) for c in r["clips"]))
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    done = errs = 0
    with open(f"{QA}/clipprep.jsonl", "a") as fh, ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(prep, r) for r in recs]):
            r = f.result()
            fh.write(json.dumps(r) + "\n"); fh.flush()
            done += 1; errs += bool(r.get("error"))
            if done % 25 == 0:
                print(f"{done}/{len(recs)} errors={errs}", flush=True)
    print(f"DONE {done} errors={errs}", flush=True)
