#!/usr/bin/env python3
"""Export the repaired segment questions plus their clips as one self-contained bundle.

Only the distractor rebuild runs on the Gemini quota. The two gates that follow --
blind-guess voting and the visual re-check -- are left for a local model, so this
writes what those gates need and nothing else: one question per record, the 480p clip
that shows it, and the evidence the answer came from.

Gate order matters and is easy to get wrong. Shuffle BEFORE measuring guessability:
generators favour putting the answer first and blind models favour picking first, so
testing an unshuffled set reports a guess rate that is too high for the wrong reason.
These questions are already shuffled (deterministically, seeded on the question text).

Usage:  export_segment_bundle.py [--repaired pilot.json] [--out bundle/]
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

REPAIR_ROOT = Path("/mnt/data/data_anno/runningbench_segment_repair")
CLIPS = REPAIR_ROOT / "clips_480p"
SEG_ROOT = "/mnt/data/data_anno/runningbench_qa_expansion/annotations_60s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repaired", default=str(REPAIR_ROOT / "repaired_all.json"))
    ap.add_argument("--out", default=str(REPAIR_ROOT / "bundle"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    repaired = json.load(open(a.repaired))["repaired"]

    # Segment metadata, keyed by the file the question came from.
    meta = {}
    for path in glob.glob(f"{SEG_ROOT}/**/*.json", recursive=True):
        try:
            rec = json.load(open(path))
        except Exception:
            continue
        raw = rec.get("original_video_path") or ""
        stem = re.sub(r"_segment_\d+$", "", os.path.basename(raw).rsplit(".", 1)[0])
        meta[path] = {"video": stem, "segment_index": rec.get("segment_index"),
                      "start_time_sec": rec.get("start_time_sec"),
                      "end_time_sec": rec.get("end_time_sec"),
                      "evidence": rec.get("dense_annotations") or rec.get("annotation_raw")}

    rows, missing_clip = [], 0
    for r in repaired:
        m = meta.get(r["file"])
        if not m:
            continue
        clip = CLIPS / m["video"] / f"{m['video']}_segment_{m['segment_index']}.480p.mp4"
        if not clip.exists():
            missing_clip += 1
            clip = None
        item = r["item"]
        rows.append({
            "id": f"{m['video']}_seg{m['segment_index']}_q{r['index']}",
            "video": m["video"],
            "segment_index": m["segment_index"],
            "span_sec": [m["start_time_sec"], m["end_time_sec"]],
            "clip": str(clip) if clip else None,
            "question": item["question"],
            "options": item["options"],
            "answer": item["answer"],
            "arity": "single" if len(item["answer"]) == 1 else "multi",
            "type": item.get("type"),
            "option_order": "shuffled",
            "distractors_rebuilt": item.get("distractors_rebuilt", 0),
            "evidence": m["evidence"],
        })

    json.dump({"n": len(rows),
               "note": "distractors rebuilt with Gemini; blind-guess and visual re-check gates NOT run",
               "questions": rows},
              open(out / "questions.json", "w"), ensure_ascii=False, indent=1)

    fmt = collections.Counter((r["arity"], len(r["options"])) for r in rows)
    with open(out / "README.md", "w") as fh:
        fh.write(f"""# 段级题修复包

{len(rows)} 道题，干扰项已用 Gemini 重造并逐条验证「被证据明确否定」。
**盲猜闸门和回看闸门都没跑** —— 留给本地模型。

## 内容

- `questions.json` — 每题一条记录，含 `clip` 指向该段的 480p 片段
- 片段在 `{CLIPS}`，1,332 段，480p/crf30

## 字段

| 字段 | 说明 |
|---|---|
| `clip` | 该题对应的 60 秒片段路径，回看闸门用 |
| `evidence` | 出题时用的结构化标注，验证干扰项用 |
| `options` / `answer` | **已打乱**，answer 是字母列表 |
| `distractors_rebuilt` | 本次重造并通过验证的干扰项个数 |

## 建议的闸门顺序

1. **盲猜 ×3**：只给 question + options，不给视频，答三次；≥2 次命中即淘汰
2. **回看**：把 `clip` 和题目一起给模型，问画面是否支持 `answer`；只留 supported

顺序不能颠倒 —— 选项已经打乱过了，直接跑盲猜即可。若你重新打乱，务必打乱后再测。

## 已知基线（同一批语料，修复前）

| | 可猜率 |
|---|---|
| 单选（主体 6 选项） | 64.7% |
| 多选（主体 8 选项） | 25.3% |
| 整体 | 48.6%（801 道抽样，95%CI [45.1, 52.0]） |

修复后应显著低于这些数字，否则说明重造没起作用。

## 格式分布

""" + "\n".join(f"- {k[0]} {k[1]} 选项: {v}" for k, v in sorted(fmt.items())) + "\n")

    print(f"wrote {len(rows)} questions -> {out}/questions.json")
    print(f"  formats: {dict(fmt)}")
    if missing_clip:
        print(f"  WARNING: {missing_clip} questions have no clip on disk")


if __name__ == "__main__":
    sys.exit(main())
