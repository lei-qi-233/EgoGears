#!/usr/bin/env python3
"""自检：在下载视频之前先确认这个包本身是完整、自洽的。

检查项：
  1. 四个数据文件存在且行数正确
  2. 每题的 (video, span_sec) 都能在片段清单里找到
  3. 每个片段都有对应的描述
  4. answer 的字母都在 options 内，且 len(answer) 合法
  5. 所有题的 screening_status 都是 passed_blind_gate
  6. ffmpeg 是否可用

用法:  python3 00_verify_kit.py
"""
import json, shutil, subprocess, sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
EXPECT = {"questions/segments_60s.jsonl": 6770,
          "metadata/segments.jsonl": 1400,
          "captions/segments_60s.jsonl": 1400}
fail = []

def chk(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond: fail.append(msg)

print("1) 文件与行数")
for rel, n in EXPECT.items():
    p = KIT / rel
    got = sum(1 for _ in p.open()) if p.exists() else -1
    chk(got == n, f"{rel}: {got} 行（应为 {n}）")
vids = json.load(open(KIT / "metadata/videos.json"))
chk(len(vids) == 126, f"metadata/videos.json: {len(vids)} 段录像（应为 126）")

qs = [json.loads(l) for l in (KIT / "questions/segments_60s.jsonl").open()]
segs = {s["segment_id"]: s for s in
        (json.loads(l) for l in (KIT / "metadata/segments.jsonl").open())}
caps = {(c["video"], tuple(c["span_sec"])) for c in
        (json.loads(l) for l in (KIT / "captions/segments_60s.jsonl").open())}

print("\n2) 题目 → 片段 对应")
segkey = {(s["video"], (s["start_sec"], s["end_sec"])) for s in segs.values()}
orphan = [q["id"] for q in qs if (q["video"], tuple(q["span_sec"])) not in segkey]
chk(not orphan, f"所有题都能定位到片段（孤儿 {len(orphan)}）")
chk(sum(s["n_questions"] for s in segs.values()) == len(qs),
    "片段清单里的题数合计与题库一致")

print("\n3) 片段 → 描述 对应")
nocap = [k for k in segkey if k not in caps]
chk(not nocap, f"每个片段都有描述（缺 {len(nocap)}）")

print("\n4) 选项与答案自洽")
bad_letter = [q["id"] for q in qs if not set(q["answer"]) <= set(q["options"])]
bad_n = [q["id"] for q in qs if q["n_options"] != len(q["options"])]
bad_k = [q["id"] for q in qs if not (1 <= len(q["answer"]) < len(q["options"]))]
chk(not bad_letter, f"answer 字母都在 options 内（异常 {len(bad_letter)}）")
chk(not bad_n, f"n_options 与 options 数量一致（异常 {len(bad_n)}）")
chk(not bad_k, f"答案个数合法（异常 {len(bad_k)}）")

print("\n5) 闸门状态")
st = {q.get("screening_status") for q in qs}
chk(st == {"passed_blind_gate"}, f"全部为 passed_blind_gate（实际 {st}）")

print("\n6) 环境")
ff = shutil.which("ffmpeg")
chk(bool(ff), f"ffmpeg: {ff or '未找到 —— 切片需要它'}")

print("\n" + ("✅ 自检通过，可以开始下载视频（见 README 步骤 2）"
              if not fail else f"❌ {len(fail)} 项未通过"))
sys.exit(1 if fail else 0)
