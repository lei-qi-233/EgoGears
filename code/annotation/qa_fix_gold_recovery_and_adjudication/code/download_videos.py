#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 0 步：下载校验所需的原始视频。

只下载校验集实际引用的 148 条录像（125.9 GiB），不是整个数据集。清单
video_manifest.json 里记录了每条录像在 conductor 上的确切位置和大小 ——
包里带的这份清单是生成校验集时同步产出的，两者必然一致。

也支持只下某一部分：先用 --source 看你要校验的来源需要哪些视频，
按需下载能省大量时间（比如只校验 p01ma_hf 只需约三分之一的量）。

用法：
  python3 download_videos.py --dest ./videos                 # 全部
  python3 download_videos.py --dest ./videos --source p01ma_gdrive
  python3 download_videos.py --dest ./videos --check         # 只检查不下载

前置：aws cli（conductor 用 S3 协议）；下载完成后 --check 会顺带验证 ffmpeg。
断点续传：重跑同一命令，已完整的文件自动跳过（按字节数比对）。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def needed_recordings(source_filter):
    need = set()
    for line in open(os.path.join(HERE, "review_set.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if source_filter and r["source"] != source_filter:
            continue
        for e in r["evidence"]:
            need.add(e["video"])
    return need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="视频保存目录")
    ap.add_argument("--source", default=None,
                    help="只下某来源需要的视频: fullvideo / excerpt / segment_screened / "
                         "p01ma_gdrive / p01ma_hf")
    ap.add_argument("--check", action="store_true", help="只检查完整性，不下载")
    a = ap.parse_args()

    man = json.load(open(os.path.join(HERE, "video_manifest.json"), encoding="utf-8"))
    need = needed_recordings(a.source)
    plan = {v: man["recordings"][v] for v in sorted(need) if v in man["recordings"]}
    absent = sorted(need - set(plan))
    total = sum(p["bytes"] for p in plan.values())
    print(f"需要 {len(plan)} 条录像，合计 {total / 2**30:.1f} GiB"
          + (f"（来源过滤: {a.source}）" if a.source else ""))
    if absent:
        print(f"⚠️ 清单里没有的录像（涉及的题会被工具跳过）: {absent}")

    os.makedirs(a.dest, exist_ok=True)
    # 递归找已有文件：校验工具就是按文件名递归匹配的，这里保持同一逻辑 ——
    # 无论视频是本脚本平铺下载的，还是用户按自己的目录结构手动放的，都算数。
    existing = {}
    for dirpath, _, files in os.walk(a.dest):
        for fn in files:
            stem = os.path.splitext(fn)[0]
            existing.setdefault(stem, os.path.join(dirpath, fn))
    have, todo = [], []
    for v, p in plan.items():
        found = existing.get(v)
        if found and os.path.getsize(found) == p["bytes"]:
            have.append(v)
        else:
            todo.append((v, p, os.path.join(a.dest, os.path.basename(p["s3_key"]))))
    print(f"已就绪 {len(have)} · 待下载 {len(todo)}"
          f"（{sum(p['bytes'] for _, p, _ in todo) / 2**30:.1f} GiB）")

    if a.check or not todo:
        ff = shutil.which("ffmpeg")
        print(f"ffmpeg: {'✅ ' + ff if ff else '❌ 未找到 —— 校验工具需要它，请安装或用 --ffmpeg 指定'}")
        if not todo:
            print("视频齐全。下一步：\n  python3 manual_verify.py --video-root "
                  f"{a.dest} --annotator 你的名字 --sample 120")
        return 0

    if not shutil.which("aws"):
        sys.exit("需要 aws cli。装好后重试；或按 README 用其他方式获取视频后 --check 验证。")
    for i, (v, p, dst) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {v}  ({p['bytes'] / 2**20:.0f} MiB)")
        r = subprocess.run(["aws", "s3", "cp", "--endpoint-url", man["endpoint"],
                            "--only-show-errors",
                            f"s3://{man['bucket']}/{p['s3_key']}", dst])
        if r.returncode != 0:
            print(f"  !! 下载失败，稍后重跑本命令续传")
    bad = [v for v, p, dst in todo
           if not (os.path.exists(dst) and os.path.getsize(dst) == p["bytes"])]
    if bad:
        print(f"\n{len(bad)} 条未完成（重跑本命令续传）: {bad[:5]}")
        return 1
    print("\n全部就绪。下一步：\n  python3 manual_verify.py --video-root "
          f"{a.dest} --annotator 你的名字 --sample 120")
    return 0


if __name__ == "__main__":
    sys.exit(main())
