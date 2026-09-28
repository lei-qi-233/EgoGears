#!/usr/bin/env python3
"""Pull the frames a Gemini verdict cites, so a non-Gemini pair of eyes can check it.

    python3 peek.py <review_id> [CLIP_LABEL MM:SS] ...
    python3 peek.py <review_id> --grid            # 9 frames spread over every clip

Both gold and the Gemini answer are printed but the frames are the point: the whole
corpus was generated, questioned and gated by one model family, so a verdict that is
only ever checked by that family is self-confirmation.
"""
import json, os, subprocess, sys

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
OUT = f"{QA}/peek"


def load(rid):
    rec = next(json.loads(l) for l in open(f"{QA}/master_index.jsonl")
               if json.loads(l)["review_id"] == rid)
    prep = {}
    for l in open(f"{QA}/clipprep.jsonl"):
        r = json.loads(l)
        prep[r["review_id"]] = r
    return rec, prep[rid]


def frame(src, t, dst):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-threads", "1",
                    "-ss", str(t), "-i", src, "-frames:v", "1",
                    "-vf", "scale=-2:480", "-q:v", "3", dst],
                   check=True, capture_output=True, timeout=300)
    return dst


def secs(ts):
    ts = ts.strip().split("-")[0].strip()
    if ":" in ts:
        m, s = ts.split(":")[-2:]
        return int(m) * 60 + float(s)
    return float(ts)


def main():
    rid = sys.argv[1]
    rec, prep = load(rid)
    os.makedirs(f"{OUT}/{rid}", exist_ok=True)
    print(f"[{rec['question_type']}] pick {rec['n_select']} of {len(rec['options'])}")
    print("Q:", rec["question"])
    for l in sorted(rec["options"]):
        print(f"   {l}. {rec['options'][l]}")
    print("gold:", rec["gold"], f"({rec['gold_provenance']})")
    for h in rec["human"]:
        print(f"human[{h['annotator']}] {h['verdict']} -> {h['final_answer']}  {h['notes'][:160]}")
    files = {c["label"]: c["file"] for c in prep["clips"]}
    args = sys.argv[2:]
    shots = []
    if not args or args[0] == "--grid":
        per = max(1, 9 // max(1, len(files)))
        for lab, f in files.items():
            dur = 60.0
            for c in rec["clips"]:
                if c["label"] == lab:
                    import re
                    m = re.search(r"_(\d+)_(\d+)\.mp4$", c["path"])
                    if m:
                        dur = float(m.group(2)) - float(m.group(1))
            for i in range(per + 1):
                shots.append((lab, dur * (i + 0.5) / (per + 1)))
    else:
        for i in range(0, len(args), 2):
            shots.append((args[i], secs(args[i + 1])))
    for lab, t in shots:
        if lab not in files:
            print("no such clip", lab, "have", list(files))
            continue
        dst = f"{OUT}/{rid}/{lab}_{t:.0f}.jpg"
        try:
            frame(files[lab], t, dst)
            print("FRAME", dst)
        except Exception as exc:
            print("fail", lab, t, exc)


if __name__ == "__main__":
    main()
