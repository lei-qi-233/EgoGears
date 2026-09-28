#!/usr/bin/env python3
"""Make questions with more than 10 clips submittable.

Every Gemini model on Floodgate rejects a request carrying more than 10 video files
("it has 12 video files but the model only supports up to 10"). Three cross_video
questions in this corpus have 11 or 12 clips, so their video review could never run --
they were sitting in the corpus as `not_reviewed`, which reads like "nobody got to it"
rather than "this always fails".

Fix: concatenate clips until the count fits, preferring clips that already SHARE a label
(cross_video shows one recording as several excerpts, all under that recording's letter,
so merging two excerpts of VIDEO_O back into one VIDEO_O loses nothing), then the
shortest. The merged file's label spells out the boundaries inside it so an answer can
still cite which excerpt and when. Other clips keep their exact labels.
"""
import json, os, re, subprocess, sys

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
CACHE = f"{QA}/clipcache"
LIMIT = 10


def duration(path):
    out = subprocess.run(["ffmpeg", "-i", path], capture_output=True, timeout=120).stderr.decode("utf-8", "replace")
    m = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", out)
    return float(m.group(1)) * 3600 + float(m.group(2)) * 60 + float(m.group(3)) if m else 60.0


def mmss(t):
    return f"{int(t)//60:02d}:{int(t)%60:02d}"


BUDGET = 13 * 1024 * 1024


def merge(files, dst, kbps=None):
    """Re-encode rather than stream-copy: the clips come from two different pipelines
    (passthrough bundle clips and 5 fps transcodes) and do not share encoder settings."""
    args = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "2"]
    for f in files:
        args += ["-i", f]
    n = len(files)
    filt = "".join(f"[{i}:v:0]scale=-2:480,fps=5,setsar=1[v{i}];" for i in range(n))
    filt += "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[out]"
    rate = (["-b:v", f"{kbps}k", "-maxrate", f"{int(kbps*1.3)}k", "-bufsize", f"{int(kbps*2)}k"]
            if kbps else ["-crf", "30"])
    args += ["-filter_complex", filt, "-map", "[out]", "-c:v", "libx264", "-preset", "veryfast"]
    args += rate + ["-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", dst]
    subprocess.run(args, check=True, capture_output=True, timeout=1800)
    return os.path.getsize(dst)


def main():
    prep = {}
    for l in open(f"{QA}/clipprep.jsonl"):
        if l.strip():
            r = json.loads(l)
            prep[r["review_id"]] = r
    over = {k: v for k, v in prep.items() if not v.get("error") and len(v.get("clips", [])) > LIMIT}
    print(f"over the {LIMIT}-video cap: {len(over)}")
    out = []
    for rid, p in over.items():
        clips = list(p["clips"])
        # key by file, not label: cross_video repeats a label across excerpts, and a
        # label-keyed table silently collapses them and mis-times the merged manifest
        durs = {c["file"]: duration(c["file"]) for c in clips}
        bylabel = {}
        for i, c in enumerate(clips):
            bylabel.setdefault(c["label"], []).append(i)
        # Collapse whole label groups, biggest group first, until the count fits. Merging a
        # group is lossless in meaning -- VIDEO_A's two excerpts are both VIDEO_A -- whereas
        # merging ACROSS labels would put two different recordings in one file under a
        # compound label, which is exactly the addressability the question needs.
        groups, n = [], len(clips)
        for lab, ii in sorted(bylabel.items(), key=lambda kv: -len(kv[1])):
            if n <= LIMIT:
                break
            if len(ii) > 1:
                groups.append(ii)
                n -= len(ii) - 1
        if n > LIMIT:
            print(f"  {rid}: cannot reach {LIMIT} by collapsing label groups ({n} left), skipped")
            continue
        picked = {i for g in groups for i in g}
        rest = [c for i, c in enumerate(clips) if i not in picked]
        os.makedirs(f"{CACHE}/{rid}", exist_ok=True)
        merged_clips = []
        for gi, g in enumerate(groups):
            grp = [clips[i] for i in g]
            dst = f"{CACHE}/{rid}/MERGED_{gi}.mp4"
            size = merge([c["file"] for c in grp], dst)
            t, spans = 0.0, []
            for c in grp:
                spans.append(f"{mmss(t)}-{mmss(t + durs[c['file']])}")
                t += durs[c["file"]]
            lab = grp[0]["label"]
            merged_clips.append({"label": f"{lab} ({len(grp)} excerpts back to back: {', '.join(spans)})",
                                 "file": dst, "bytes": size,
                                 "merged_from": [c["label"] for c in grp]})
        allc = rest + merged_clips
        total = sum(c["bytes"] for c in allc)
        # the merge re-encodes, so it can come back bigger than the pieces it replaced;
        # squeeze the merged files back under the inline budget before shipping
        for mc in merged_clips:
            if total <= BUDGET:
                break
            dur = sum(durs[clips[i]["file"]] for g in groups for i in g)
            kbps = max(100, int((BUDGET - (total - mc["bytes"])) * 0.9 * 8 / 1000 / max(1.0, dur)))
            before = mc["bytes"]
            mc["bytes"] = merge([c["file"] for c in [clips[i] for i in groups[merged_clips.index(mc)]]],
                                mc["file"], kbps)
            total = total - before + mc["bytes"]
        rec = dict(p, mode="merged", clips=allc, total_bytes=total)
        print(f"  {rid}: {len(clips)} -> {len(allc)} clips, merged groups "
              f"{[[clips[i]['label'] for i in g] for g in groups]}, {total/1048576:.1f} MB")
        out.append(rec)
    with open(f"{QA}/clipprep.jsonl", "a") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(out)} rewritten clipprep rows")


if __name__ == "__main__":
    main()
