# -*- coding: utf-8 -*-
"""按 FLOODGATE_ANNOTATION.md §4.2 切 60s 片段: 480p / libx264 crf30 / 去音轨。

不烧 MM:SS 时钟: 题面时间戳经核实是片段内相对时间, 片段从 0 开始即已对齐;
且本机 ffmpeg 未编 drawtext(libfreetype), 烧不了。
截断片段(末尾窗口超出视频时长)只切到视频真实结尾。
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

VMAP = json.load(open('/mnt/data/cvhci_video_understanding/video_map.json'))
DUR = json.load(open(os.environ.get(
    'RB_DUR', '/mnt/tmp/claude-0/-mnt-task-runtime/530b5738-50c5-4511-a092-b0779eaf18d9/scratchpad/dur.json')))
MAXB = 20 * 1024 * 1024


def cut(seg, outdir):
    vid, (st, en) = seg
    en = min(en, DUR[vid])
    dst = os.path.join(outdir, f'{vid}__{st:.0f}_{en:.0f}.mp4')
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    for crf in (30, 34, 38):                      # 太大只加 crf, 不降分辨率
        r = subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(st), '-i', VMAP[vid],
             '-t', str(max(0.5, en - st)), '-vf', 'scale=-2:480',
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf),
             '-an', '-movflags', '+faststart', dst],
            capture_output=True, text=True)
        if r.returncode != 0:
            return ('ERR', vid, r.stderr[-200:])
        if os.path.getsize(dst) <= MAXB:
            return dst
    return dst


if __name__ == '__main__':
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    segs = sorted({(r['video'], tuple(r['span_sec']))
                   for r in (json.loads(l) for l in open(src))})
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda s: cut(s, outdir), segs))
    err = [r for r in res if isinstance(r, tuple)]
    ok = [r for r in res if not isinstance(r, tuple)]
    for e in err[:5]:
        print('  失败:', e)
    if ok:
        sizes = [os.path.getsize(p) for p in ok]
        print(f'{len(ok)}/{len(segs)} 片段已切  平均 {sum(sizes)/len(sizes)/1e6:.1f}MB  '
              f'最大 {max(sizes)/1e6:.1f}MB  失败 {len(err)}')
    else:
        print(f'全部失败 ({len(segs)} 片段)')
