#!/usr/bin/env python3
"""Re-cut the 60-second clips the segment corpus needs for a visual re-check gate.

The segment questions were generated from captions and never re-checked against the
footage -- the gap that lets a blind model score 48.6% on them. Adding the gate needs
the clips back; the originals were cleaned up except for 276 of them.

Cut at 480p/crf30, matching the full-video pipeline's re-check encode. That is sharper
than the 288p global pass on purpose: a re-check at 360p once misread a legible street
sign as blurred and produced a false "contradicted".

Local ffmpeg only, no API. Resumable -- an existing clip is left alone.

Usage:  cut_segment_recheck_clips.py [--jobs 6] [--limit N]
"""
import argparse
import collections
import concurrent.futures
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SEG_ROOT = "/mnt/data/data_anno/runningbench_qa_expansion/annotations_60s"
SCENES = "/mnt/data/cvhci_video_understanding/metadata/route_manifest/scene_manifest.json"
FFMPEG = "/mnt/data/data_anno/ffmpeg-7.0.2-amd64-static/ffmpeg"
OUT = Path("/mnt/data/data_anno/runningbench_segment_repair/clips_480p")


def plan():
    src = {r["run_id"]: r["file"] for r in json.load(open(SCENES))["runs"]}
    jobs, unmatched = [], set()
    for path in sorted(glob.glob(f"{SEG_ROOT}/**/*.json", recursive=True)):
        try:
            rec = json.load(open(path))
        except Exception:
            continue
        raw = rec.get("original_video_path") or ""
        stem = re.sub(r"_segment_\d+$", "", os.path.basename(raw).rsplit(".", 1)[0])
        if stem not in src:
            unmatched.add(stem)
            continue
        start = float(rec.get("start_time_sec", 0))
        end = float(rec.get("end_time_sec", start + 60))
        jobs.append((stem, src[stem], start, end - start, int(rec.get("segment_index", 0))))
    return jobs, unmatched


def cut(job):
    stem, source, start, length, index = job
    dst = OUT / stem / f"{stem}_segment_{index}.480p.mp4"
    if dst.exists() and dst.stat().st_size > 4096:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([FFMPEG, "-y", "-ss", str(start), "-i", source, "-t", str(length),
                        "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "30", "-an", "-movflags", "+faststart", str(dst)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "ok" if r.returncode == 0 and dst.exists() else "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    jobs, unmatched = plan()
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} clips from {len({j[0] for j in jobs})} sources"
          + (f"; {len(unmatched)} unmatched stems: {sorted(unmatched)}" if unmatched else ""), flush=True)

    tally = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for i, outcome in enumerate(pool.map(cut, jobs), 1):
            tally[outcome] += 1
            if i % 100 == 0 or i == len(jobs):
                size = sum(f.stat().st_size for f in OUT.rglob("*.mp4")) / 2**30
                print(f"  {i}/{len(jobs)}  {dict(tally)}  {size:.1f} GiB", flush=True)
    print(f"\ndone: {dict(tally)} -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
