#!/usr/bin/env python3
"""Cut the 180-second windows at 480p so questions built on them can be re-checked.

The 180s annotations are aggregations of three 60s captions -- `derived_from` names
them -- not independent viewings. So every question written from them inherits whatever
the 60s pass got wrong, and the only thing that catches that is showing a model the
actual footage. These clips are that gate's input.

480p, matching the full-video pipeline: a re-check at 360p once read a legible street
sign as blurred and returned a false "contradicted". The re-check must see more detail
than the pass that wrote the annotation, never less.

Local ffmpeg only. Resumable.

Usage:  cut_window_clips.py [--jobs 6]
"""
import argparse
import collections
import concurrent.futures
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = "/mnt/data/data_anno/runningbench_gap_repair/annotations_180s"
FFMPEG = "/mnt/data/data_anno/ffmpeg-7.0.2-amd64-static/ffmpeg"
OUT = Path("/mnt/data/data_anno/runningbench_windows/clips_480p")
INLINE_CAP = 11 * 1024 * 1024


def plan():
    jobs = []
    for path in sorted(glob.glob(f"{SRC}/**/*.json", recursive=True)):
        rec = json.load(open(path))
        source = rec.get("original_video_path")
        if not source or not os.path.exists(source):
            continue
        stem = os.path.basename(source).rsplit(".", 1)[0]
        start = float(rec["start_time_sec"])
        length = float(rec["end_time_sec"]) - start
        jobs.append((stem, int(rec.get("chunk_index", 0)), source, start, length))
    return jobs


def cut(job):
    stem, idx, source, start, length = job
    dst = OUT / stem / f"{stem}_w{idx}.480p.mp4"
    if dst.exists() and dst.stat().st_size > 4096:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    # The API takes video inline up to 11 MiB, and crf-only encoding of a 180 s window
    # lands around 12-18 MiB on these high-motion recordings -- half the clips would be
    # rejected. Cap the bitrate to fit instead, keeping 480p: detail per frame matters
    # more to the re-check than bitrate does, and 450 kbps is still several times what
    # the 288p global annotation pass runs at.
    budget = int(INLINE_CAP * 8 * 0.88 / max(length, 1) / 1000)
    r = subprocess.run([FFMPEG, "-y", "-ss", str(start), "-i", source, "-t", str(length),
                        "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast",
                        "-b:v", f"{budget}k", "-maxrate", f"{int(budget * 1.3)}k",
                        "-bufsize", f"{budget * 2}k",
                        "-an", "-movflags", "+faststart", str(dst)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode == 0 and dst.exists() and dst.stat().st_size > INLINE_CAP:
        return "oversize"
    return "ok" if r.returncode == 0 and dst.exists() else "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    jobs = plan()
    print(f"{len(jobs)} windows from {len({j[0] for j in jobs})} videos", flush=True)
    tally = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for i, outcome in enumerate(pool.map(cut, jobs), 1):
            tally[outcome] += 1
            if i % 20 == 0 or i == len(jobs):
                size = sum(f.stat().st_size for f in OUT.rglob("*.mp4")) / 2**30
                print(f"  {i}/{len(jobs)}  {dict(tally)}  {size:.2f} GiB", flush=True)
    print(f"done: {dict(tally)} -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
