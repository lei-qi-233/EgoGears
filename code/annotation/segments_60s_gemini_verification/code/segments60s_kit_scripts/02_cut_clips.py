#!/usr/bin/env python3
"""按 metadata/segments.jsonl 把源录像切成 1,400 个评测片段。

切 1,400 次而不是 6,770 次 —— 每个片段平均被 4.8 道题共用，按片段切一次即可。

编码参数是调出来的，**不要随手改**：

  -vf scale=-2:480   480p 是下限。实测降到 360p 会让模型把清晰可读的路牌报成
                     "blurry"，判出假的否定，误杀本来正确的题。宁可牺牲时间分辨率
                     也不牺牲空间分辨率。
  -crf 30            片段过大时只提 crf（30→34→38），绝不降分辨率。
  -an                **必须去音轨**。P01 有音频、P02 无、P03 有 —— 保留音轨等于
                     给模型留了一个完美的"参与者"捷径，测出来的不是视频理解。
  +faststart         moov 前置，便于流式读取。

断点续跑：已存在且非空的片段会跳过。

用法:
  python3 02_cut_clips.py --video-map video_map.json --out clips/ [--jobs 8]
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def cut(ff, src, start, dur, dst, crf=30, fps=None):
    vf = f"scale=-2:480{',fps=' + str(fps) if fps else ''}"
    cmd = [ff, "-y", "-v", "error", "-ss", f"{start}", "-i", str(src), "-t", f"{dur}",
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-map", default="video_map.json")
    ap.add_argument("--kit", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--out", default="clips")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--max-mb", type=float, default=15.0,
                    help="单片段字节上限；超了自动提 crf，再超降帧率")
    a = ap.parse_args()

    vmap = json.load(open(a.video_map))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    segs = [json.loads(l) for l in open(f"{a.kit}/metadata/segments.jsonl")]
    todo = [s for s in segs if s["video"] in vmap]
    skipped = [s for s in segs if s["video"] not in vmap]
    print(f"片段 {len(segs)} 个，可切 {len(todo)}，缺录像跳过 {len(skipped)}")

    stats = {"ok": 0, "skip": 0, "shrunk": 0, "fail": 0}

    def work(s):
        dst = out / f"{s['segment_id']}.mp4"
        if dst.exists() and dst.stat().st_size > 4096:
            stats["skip"] += 1; return
        dur = s["end_sec"] - s["start_sec"]
        try:
            cut(a.ffmpeg, vmap[s["video"]], s["start_sec"], dur, dst)
            for crf, fps in ((34, None), (38, None), (38, 15)):
                if dst.stat().st_size <= a.max_mb * 1e6: break
                cut(a.ffmpeg, vmap[s["video"]], s["start_sec"], dur, dst, crf, fps)
                stats["shrunk"] += 1
            stats["ok"] += 1
        except subprocess.CalledProcessError as e:
            stats["fail"] += 1
            print(f"  失败 {s['segment_id']}: {e.stderr.decode()[:120]}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(work, todo))
    print(f"\n完成: 新切 {stats['ok']}，跳过(已存在) {stats['skip']}，"
          f"重编码降码率 {stats['shrunk']}，失败 {stats['fail']}")
    if skipped:
        print(f"因缺录像未切的片段 {len(skipped)} 个，涉及录像: "
              f"{sorted({s['video'] for s in skipped})[:5]} …")


if __name__ == "__main__":
    main()
