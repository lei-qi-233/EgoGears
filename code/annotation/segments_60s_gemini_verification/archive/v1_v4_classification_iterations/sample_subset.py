# -*- coding: utf-8 -*-
"""按题型分层抽取子集, 保证 9 个报告类全部出现。

分配: report_class 层面按比例 + 下限 min(FLOOR, 类题数) + 从最大类回收以凑齐总数;
      类内按 arity 比例细分。
约束: 每个 60s 片段最多 1 题; 每个视频最多 RB_VIDEO_CAP 题 (实测 5 使设计效应最低)。
比例分配保证整体分数无偏; 触发下限的类需按 weight 字段加权才能还原全量占比。
"""
import json, random, collections, os

SRC=os.environ.get('RB_TYPED','segments_60s_typed.jsonl')
TARGET=int(os.environ.get('RB_TARGET','677'))
FLOOR=int(os.environ.get('RB_FLOOR','20'))        # 每个报告类的最少题数
VCAP=int(os.environ.get('RB_VIDEO_CAP','5'))
SEED=int(os.environ.get('RB_SEED','20260916'))
OUT=os.environ.get('RB_OUT','segments_60s_subset.jsonl')

rows=[json.loads(l) for l in open(SRC)]
# 排除视频不可看的题: span 超出视频真实结尾, 或实际可见 <15s (含 kit 官方短片段)
EXCL=os.environ.get('RB_EXCLUDE','excluded_short_clips.json')
if os.path.exists(EXCL):
    bad=set(json.load(open(EXCL)))
    before=len(rows); rows=[r for r in rows if r['id'] not in bad]
    print(f'排除视频过短/被截断的题 {before-len(rows)} 条, 剩余池 {len(rows)}')
N=len(rows)
by_cls=collections.defaultdict(list)
for r in rows: by_cls[r['report_class']].append(r)

# --- 类级配额: 比例 -> 抬下限 -> 从最大类等比回收 ---
quota={k:max(min(FLOOR,len(v)), round(len(v)*TARGET/N)) for k,v in by_cls.items()}
over=sum(quota.values())-TARGET
floored={k for k in quota if quota[k]>round(len(by_cls[k])*TARGET/N)}
# 按配额大小等比回收, 让大类吸收绝大部分, 小类权重保持接近 1
donors=sorted((k for k in quota if k not in floored), key=lambda k:-quota[k])
tot=sum(quota[k] for k in donors)
cut={k:min(quota[k]-FLOOR, int(over*quota[k]/tot)) for k in donors}
for k,c in cut.items(): quota[k]-=c
over-=sum(cut.values())
i=0
while over>0:                      # 余数仍从最大类起扣
    k=donors[i%len(donors)]
    if quota[k]>FLOOR: quota[k]-=1; over-=1
    i+=1

rng=random.Random(SEED)
used_seg=set(); per_vid=collections.Counter(); picked=[]
for cls in sorted(by_cls, key=lambda k:-len(by_cls[k])):
    # 类内按 arity 比例细分
    sub=collections.defaultdict(list)
    for r in by_cls[cls]: sub[r['arity']].append(r)
    q={a:round(len(v)*quota[cls]/len(by_cls[cls])) for a,v in sub.items()}
    d=quota[cls]-sum(q.values())
    if d and q: q[max(q,key=lambda a:len(sub[a]))]+=d
    for a,pool in sub.items():
        pool=pool[:]; rng.shuffle(pool); got=0
        for vlim,segdedup in ((VCAP,True),(10**9,True),(10**9,False)):
            if got>=q.get(a,0): break
            for r in pool:
                if got>=q.get(a,0): break
                if r in picked: continue
                seg=(r['video'],tuple(r['span_sec']))
                if segdedup and seg in used_seg: continue
                if per_vid[r['video']]>=vlim: continue
                used_seg.add(seg); per_vid[r['video']]+=1; picked.append(r); got+=1

# 保证每个视频至少 1 题: 同 (report_class, arity) 层交换, 配额不变
allv={r['video'] for r in rows}
by_vid=collections.defaultdict(list)
for r in rows: by_vid[r['video']].append(r)
for v in sorted(allv-{r['video'] for r in picked}):
    for r in sorted(by_vid[v], key=lambda x:x['id']):
        key=(r['report_class'],r['arity'])
        cnt=collections.Counter(x['video'] for x in picked)
        outs=[o for o in picked if (o['report_class'],o['arity'])==key and cnt[o['video']]>1]
        if not outs: continue
        o=max(outs, key=lambda x:cnt[x['video']])
        picked.remove(o); picked.append(r); break

# 每个类的加权系数: 还原全量占比用
w={k:(len(by_cls[k])/N)/(quota[k]/TARGET) for k in quota}
for r in picked: r['subset_weight']=round(w[r['report_class']],4)

picked.sort(key=lambda r:(r['video'],r['span_sec'],r['id']))
with open(OUT,'w') as f:
    for r in picked: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(f'{len(picked)} 题 -> {OUT}  (目标 {TARGET}, 下限 {FLOOR}, 每视频上限 {VCAP}, 种子 {SEED})')
print('触发下限的类:', sorted(floored))
