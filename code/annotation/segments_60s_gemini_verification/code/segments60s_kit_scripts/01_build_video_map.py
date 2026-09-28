#!/usr/bin/env python3
"""把 video id 映射到本地视频文件，并核对 126 段录像是否齐全。

题目用 `video` 字段引用录像，取值是**去掉扩展名的文件名**（如 P02_Running_Day_Trial02）。
本脚本扫描你下载到的目录，建立 id -> 路径 的映射。

HuggingFace 集内存在按文件名与字节数完全相同的重复文件，脚本会按 (文件名, 大小)
检测并只保留一个，同时报告重复项。

用法:
  python3 01_build_video_map.py --roots /path/to/google_drive /path/to/huggingface
"""
import argparse, collections, hashlib, json, os, sys
from pathlib import Path

VIDEO_EXT = {".mp4", ".mov", ".MP4", ".MOV", ".m4v", ".webm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="下载目录，可给多个")
    ap.add_argument("--kit", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--out", default="video_map.json")
    a = ap.parse_args()

    found = collections.defaultdict(list)
    for root in a.roots:
        for dirpath, _, names in os.walk(root):
            for n in names:
                if Path(n).suffix in VIDEO_EXT:
                    p = Path(dirpath) / n
                    found[p.stem].append(p)

    vmap, dup = {}, []
    for vid, paths in found.items():
        if len(paths) > 1:
            sizes = {p.stat().st_size for p in paths}
            dup.append((vid, [str(p) for p in paths], len(sizes) == 1))
        vmap[vid] = str(sorted(paths, key=lambda p: (-p.stat().st_size, str(p)))[0])

    need = {v["video"] for v in json.load(open(f"{a.kit}/metadata/videos.json"))}
    have = need & set(vmap)
    miss = sorted(need - set(vmap))

    json.dump({k: vmap[k] for k in sorted(have)}, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"扫描到视频文件 {sum(len(v) for v in found.values())} 个，唯一 id {len(vmap)} 个")
    print(f"本套题需要 {len(need)} 段，已找到 {len(have)} 段 -> {a.out}")
    if dup:
        print(f"\n同名重复 {len(dup)} 组（已各取其一）:")
        for vid, paths, same in dup[:10]:
            print(f"  {vid}  {'字节数相同' if same else '字节数不同 ← 需人工确认'}")
            for p in paths: print(f"     {p}")
    if miss:
        print(f"\n缺 {len(miss)} 段录像，这些录像上的题无法评测:")
        for v in miss[:20]: print(f"  {v}")
        if len(miss) > 20: print(f"  …另 {len(miss)-20} 段")
        sys.exit(1)
    print("\n✅ 全部录像齐备")


if __name__ == "__main__":
    main()
