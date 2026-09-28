#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunningBench 一键手工校验程序。

给校验员用：你拿到 review_set.jsonl（1,285 道题）和原始视频后，运行本程序逐题走完
标准校验流程。程序会强制正确的顺序——先盲猜、锁定之后才允许看视频——这个顺序不是
形式主义：出题模型倾向把答案写得更"像真的"，人先看过视频再假装盲猜，测出来的可猜率
是假的。整套语料曾因为顺序颠倒测出过 56.6% 的虚高读数。

流程（对每道题）：
  第一步 盲猜    只看题干和选项，凭直觉答一次并锁定。
                你能答对 => 这道题不看视频也能做 => 标注失败，无论答案对不对。
  第二步 看视频  程序用 ffmpeg 现场切出该题的证据片段（480p），用 ffplay 播放，
                或告诉你文件路径自己打开。
  第三步 判定    看完后给出你的最终答案，并对标注答案下判定：
                  correct       标注答案对，且干扰项确实错
                  wrong_answer  画面显示的正确答案与标注不符（写明你认为的答案）
                  ambiguous     有第二组说得通的答案
                  cant_tell     画面判定不了
  结果逐题落盘（JSONL，可随时中断续跑）。

准备工作：
  1. 下载原始视频（三选一）：
       a) conductor:
          aws s3 sync --endpoint-url https://conductor.data.apple.com \
            s3://yuedong/cvhci_video_understanding/raw/ ./videos/
       b) Google Drive 原始采集目录（P01/P02/P03 共 54 个 mp4）
       c) HuggingFace 数据集原始目录（Alec/Mikail/Markus 共 72 个视频）
     放在任意目录下即可，子目录结构不限——程序按文件名递归匹配。
  2. 装 ffmpeg（切片段必需）；装 ffplay 可自动播放，没有也行，程序会给出片段路径。

用法：
  python3 manual_verify.py --video-root ./videos --annotator 你的名字
  python3 manual_verify.py --video-root ./videos --annotator 张三 --sample 120   # 分层抽样
  python3 manual_verify.py --video-root ./videos --annotator 张三 --source p01ma_gdrive
  中断后重跑同一命令即可续跑（已判定的题自动跳过）。

产出：
  review_results.jsonl   每题一条 human_checks 记录
  结束时打印汇总：人工与标注的一致率、人工盲猜命中率（这是最重要的指标）。
"""
import argparse
import collections
import json
import os
import random
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def build_video_index(root):
    """文件名（不含扩展名）→ 路径。按文件名匹配而不按目录，因为三个来源的目录结构
    不同，且校验员的下载方式（conductor 同步 / 手动整理）会产生不同层级。"""
    index = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in VIDEO_EXT:
                index.setdefault(stem, os.path.join(dirpath, fn))
    return index


def cut_clip(ffmpeg, source, span, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 4096:
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    start, end = span
    r = subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-ss", str(start), "-i", source,
         "-t", str(end - start), "-vf", "scale=-2:480", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "28", "-an", "-movflags", "+faststart", dst],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and os.path.exists(dst)


def ask_letters(prompt, n_expected, valid):
    """读入恰好 n 个选项字母；容忍 'ADG'、'a,d,g'、'a d g' 等写法。"""
    while True:
        raw = input(prompt).strip().upper()
        if raw in ("Q", "QUIT", "退出"):
            return None
        letters = sorted(set(re.findall(r"[A-H]", raw)))
        if len(letters) == n_expected and all(l in valid for l in letters):
            return letters
        print(f"  需要恰好 {n_expected} 个不同的选项字母（{'/'.join(sorted(valid))}），再来一次。")


def ask_choice(prompt, choices):
    keys = "/".join(str(i + 1) for i in range(len(choices)))
    for i, c in enumerate(choices):
        print(f"    [{i + 1}] {c[1]}")
    while True:
        raw = input(f"{prompt} ({keys}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1][0]
        print("  输入编号。")


def main():
    ap = argparse.ArgumentParser(description="RunningBench 手工校验")
    ap.add_argument("--review-file", default=os.path.join(HERE, "review_set.jsonl"))
    ap.add_argument("--video-root", required=True, help="原始视频所在目录（递归查找）")
    ap.add_argument("--out", default=os.path.join(HERE, "review_results.jsonl"))
    ap.add_argument("--annotator", required=True, help="你的标识，写进每条记录")
    ap.add_argument("--sample", type=int, default=0,
                    help="分层随机抽 N 道（按来源比例）；不填则全量")
    ap.add_argument("--source", default=None,
                    help="只看某一来源: fullvideo / excerpt / segment_screened / "
                         "p01ma_gdrive / p01ma_hf")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--no-play", action="store_true", help="不自动播放，只打印片段路径")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg 路径（不在 PATH 时用）")
    a = ap.parse_args()

    ffmpeg = a.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg or not os.path.exists(ffmpeg) and not shutil.which(ffmpeg):
        sys.exit("需要 ffmpeg（用于切证据片段）。装好后重试，或用 --ffmpeg 指定路径。")
    ffplay = None if a.no_play else shutil.which("ffplay")

    rows = [json.loads(l) for l in open(a.review_file, encoding="utf-8")]
    if a.source:
        rows = [r for r in rows if r["source"] == a.source]
    if a.sample:
        rng = random.Random(a.seed)
        by = collections.defaultdict(list)
        for r in rows:
            by[r["source"]].append(r)
        picked = []
        for k in sorted(by):
            g = by[k]
            rng.shuffle(g)
            picked += g[:max(1, round(a.sample * len(g) / len(rows)))]
        rng.shuffle(picked)
        rows = picked[:a.sample]

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["review_id"])
            except Exception:
                pass
    todo = [r for r in rows if r["review_id"] not in done]

    index = build_video_index(a.video_root)
    missing = sorted({e["video"] for r in todo for e in r["evidence"]} - set(index))
    print(f"待校验 {len(todo)} 道（已完成 {len(done)}，本次范围 {len(rows)}）")
    print(f"视频索引: {len(index)} 个文件")
    if missing:
        print(f"⚠️  {len(missing)} 条录像在 --video-root 下找不到，"
              f"涉及的题会跳过: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    print("随时输入 q 退出；重跑同一命令续跑。\n" + "=" * 62)

    clips_dir = os.path.join(HERE, "review_clips")
    out = open(a.out, "a", encoding="utf-8")
    stats = collections.Counter()

    for i, r in enumerate(todo, 1):
        if any(e["video"] not in index for e in r["evidence"]):
            continue
        letters = sorted(r["options"])
        n = len(r["answer"])
        print(f"\n[{i}/{len(todo)}] {r['source']} · {r['unit']} · {r['question_type']}")
        print(f"Q: {r['question']}\n")
        for L in letters:
            print(f"   {L}. {r['options'][L]}")

        # 第一步 · 盲猜（锁定后才展示视频；这一步做完之前程序绝不打印片段路径）
        blind = ask_letters(f"\n第一步·盲猜（不看视频，选 {n} 个字母）: ", n, set(letters))
        if blind is None:
            break
        blind_hit = blind == r["answer"]

        # 第二步 · 看视频
        print("\n第二步·看视频，切片中 ...")
        clip_paths = []
        for e in r["evidence"]:
            dst = os.path.join(clips_dir, r["review_id"],
                               f"{e['label']}_{int(e['span_sec'][0])}_{int(e['span_sec'][1])}.mp4")
            if cut_clip(ffmpeg, index[e["video"]], e["span_sec"], dst):
                clip_paths.append((e["label"], dst))
            else:
                print(f"  !! 切片失败 {e['label']}（{e['video']}）")
        for label, path in clip_paths:
            if ffplay:
                print(f"  播放 {label}（关掉播放窗口继续）...")
                subprocess.run([ffplay, "-loglevel", "error", "-autoexit", path])
            else:
                print(f"  {label}: {path}")
        if not ffplay and clip_paths:
            input("  自行打开上述片段观看，看完按回车 ...")

        # 第三步 · 判定
        final = ask_letters(f"\n第三步·看完视频后你的最终答案（{n} 个字母）: ", n, set(letters))
        if final is None:
            break
        agree = final == r["answer"]
        print(f"  标注答案是 {','.join(r['answer'])} —— 你{'一致' if agree else '不一致'}。")
        verdict = ask_choice("  你的判定", [
            ("correct", "标注正确：正确项画面支持，干扰项画面否定"),
            ("wrong_answer", "标注答案错了（画面显示的是你选的）"),
            ("ambiguous", "有第二组说得通的答案 / 题干歧义"),
            ("cant_tell", "画面判定不了")])
        notes = input("  备注（可空）: ").strip()

        record = {"review_id": r["review_id"], "source": r["source"],
                  "unit": r["unit"], "question_type": r["question_type"],
                  "annotator": a.annotator,
                  "human_checks": {
                      "blind_attempt": blind, "blind_hit": blind_hit,
                      "final_answer": final, "agrees_with_gold": agree,
                      "verdict": verdict, "notes": notes},
                  "gold": r["answer"],
                  "machine_blind_hits": r.get("machine_blind_hits")}
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        stats["done"] += 1
        stats["blind_hit"] += blind_hit
        stats[f"v_{verdict}"] += 1

    out.close()
    n = stats["done"]
    if n:
        print("\n" + "=" * 62)
        print(f"本次完成 {n} 道")
        print(f"  人工盲猜命中: {stats['blind_hit']}/{n} = {100 * stats['blind_hit'] / n:.0f}%"
              f"   ← 最重要的指标，验收线是 <15%")
        for k in ("v_correct", "v_wrong_answer", "v_ambiguous", "v_cant_tell"):
            if stats[k]:
                print(f"  {k[2:]:14s} {stats[k]}")
        print(f"结果在 {a.out}")


if __name__ == "__main__":
    main()
