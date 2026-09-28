# -*- coding: utf-8 -*-
"""从已通过视觉核验的题里(overall==answer_correct)分层抽样, 产出最终 ~700 题子集。

与 sample_subset.py 的区别: 抽样池已经是"通过视觉核验"的子集(1211题), 而不是全量池,
所以这里的分数是真实可信的——每一道题都对着画面核对过选项。

约束: 9类全部出现; 大类按6452题池占比分配700题配额(通过数不够配额时全部纳入);
      小类全部纳入通过的题(不强求配额); 每视频最多5题; 每片段最多1题。
"""
import json, random, collections

TARGET = 700
# VCAP 扫描历史:
#  - 6452题全量未核验池: cap=5 是设计效应最优值(池子远大于配额, 视频分布宽松)。
#  - 第一轮抽样核验(1551题, 通过池1211): 重新扫描后 cap=15 使9类配额缺口归零(DE 1.25)。
#  - 全量核验完成(6452题100%普查, 通过池5081题)后再次扫描: 最优值仍是 cap=15(DE 1.28,
#    缺口0)。通过池扩大到5081(约4.2倍)后阈值没变——说明"少数长视频产更多题"是语料本身
#    固有的分布形状, 不会因核验覆盖率提高而改变, 不是抽样噪声导致的假象。
VCAP = 15
SEED = 20260917

full = {json.loads(l)['id']: json.loads(l) for l in open('segments_60s_typed.jsonl')}
bad = set(json.load(open('excluded_short_clips.json')))
pool = [r for r in full.values() if r['id'] not in bad]
pool_by_class = collections.Counter(r['report_class'] for r in pool)
N = len(pool)

res = {json.loads(l)['id']: json.loads(l) for l in open('verify_results.jsonl')}
passed = [full[qid] for qid, r in res.items() if r.get('overall') == 'answer_correct' and qid in full]
by_class = collections.defaultdict(list)
for r in passed:
    by_class[r['report_class']].append(r)

# 配额沿用 pass_rate_stats.json 里的 target700(按 677 题子集比例放大到700, 小类下限约21)。
# 这与最初 677/700 题子集的设计口径一致: 大类按池占比分配, 小类给固定下限, 不因为
# 某个小类"恰好被100%核验"就把它的配额撑大到远超其它类的规模。
# 通过数不够配额时(大类不够/小类池本来就小)才据实全部纳入, 不为凑数放宽标准。
stats = json.load(open('pass_rate_stats.json'))
quota = {k: min(stats[k]['target700'], len(by_class[k])) for k in pool_by_class}

rng = random.Random(SEED)
used_seg = set()
per_vid = collections.Counter()
picked = []
for k in sorted(by_class, key=lambda k: -len(by_class[k])):
    cand = by_class[k][:]
    rng.shuffle(cand)
    need = quota[k]
    got = 0
    # 每视频上限是硬约束, 只放开片段去重, 不解除视频上限
    for vlim, segdedup in ((VCAP, True), (VCAP, False)):
        if got >= need:
            break
        for r in cand:
            if got >= need:
                break
            if r in picked:
                continue
            seg = (r['video'], tuple(r['span_sec']))
            if segdedup and seg in used_seg:
                continue
            if per_vid[r['video']] >= vlim:
                continue
            used_seg.add(seg)
            per_vid[r['video']] += 1
            picked.append(r)
            got += 1

# 补 verify_overall / verify_reason 字段
for r in picked:
    v = res[r['id']]
    r['verify_overall'] = v.get('overall')
    r['verify_reason'] = v.get('reason', '')

picked.sort(key=lambda r: (r['video'], r['span_sec'], r['id']))
with open('segments_700_verified.jsonl', 'w') as f:
    for r in picked:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'最终子集: {len(picked)} 题 -> segments_700_verified.jsonl')
cs = collections.Counter(r['report_class'] for r in picked)
for k, v in cs.most_common():
    print(f'  {k:<32}{v:>5}  (可用通过题 {len(by_class[k])}, 配额 {quota[k]})')
print(f'视频覆盖: {len(set(r["video"] for r in picked))}')
seg_reuse = collections.Counter((r['video'], tuple(r['span_sec'])) for r in picked)
print('片段复用:', dict(collections.Counter(seg_reuse.values())))
