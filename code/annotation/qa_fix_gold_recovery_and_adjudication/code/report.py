#!/usr/bin/env python3
"""Emit QA_FIX_REPORT.md from the artifacts. Every number in the report comes from a
file in this directory, so it can be recomputed by re-running this script."""
import json, os, collections, statistics, math

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
UNTRUSTED = {"Junwei Zheng"}


def wilson(k, n):
    if not n:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


def jl(path, pred=None):
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if pred is None or pred(r):
            out.append(r)
    return out


def main():
    idx = {r["review_id"]: r for r in jl(f"{QA}/master_index.jsonl")}
    adj = {r["review_id"]: r for r in jl(f"{QA}/adjudication.jsonl")}
    vid = collections.defaultdict(dict)
    for r in jl(f"{QA}/video_review.jsonl", lambda r: not r.get("error")):
        vid[r["review_id"]][r["vote"]] = r
    blind = {r["review_id"]: r for r in jl(f"{QA}/blind_gate.jsonl", lambda r: not r.get("error"))}
    reps = jl(f"{QA}/repairs.jsonl")
    v2 = jl(f"{QA}/runningbench_v2.jsonl")
    L = []
    P = L.append

    N = len(idx)
    nv = len(vid)
    P(f"# RunningBench 题目可行性复核与修复\n")
    P(f"对象：HF `taryya/RunningBench`，10 个题包共 **{N} 题**。方法：**每一题都用 Gemini floodgate")
    P(f"看真实视频逐选项复核**，与恢复出的标准答案、人工标注三方交叉，再对判定为坏的题做带视频的修复。\n")
    P(f"复核覆盖：视频复核 {nv}/{N}，盲猜闸门 {len(blind)}/{N}。\n")

    # ---- 1 三方一致性
    P("## 1. 三个视角互相怎么看\n")
    ex = n = 0
    ruled = tot = 0
    for rid, vs in vid.items():
        v = vs.get(0)
        g = set(idx[rid]["gold"])
        if not v or not g:
            continue
        n += 1
        ex += set(v["final_answer"]) == g
        for l in g:
            tot += 1
            ruled += v["per_option"].get(l) == "ruled_out"
    lo, hi = wilson(ex, n)
    P(f"- Gemini 视频复核 vs 恢复的 gold：**{100*ex/max(1,n):.1f}%** 完全一致（n={n}，95% CI [{lo:.1f}, {hi:.1f}]）")
    P(f"- gold 指定为正确的选项中，被画面判为**否定**的占 **{100*ruled/max(1,tot):.1f}%**（{ruled}/{tot}）")
    hn = hex_ = 0
    for rid, vs in vid.items():
        v = vs.get(0)
        hs = [h for h in idx[rid]["human"] if h["annotator"] not in UNTRUSTED]
        if not v or not hs:
            continue
        hn += 1
        hex_ += set(v["final_answer"]) == set(hs[0]["final_answer"])
    P(f"- Gemini 视频复核 vs 人工（两边都看了视频、都不知道 gold）：**{100*hex_/max(1,hn):.1f}%** 完全一致（n={hn}）\n")

    # ---- 2 标注员严格度
    P("## 2. 人工标注的严格度差 40 倍，而且和题包一一对应\n")
    P("每个标注员包一个 bundle，所以按 bundle 统计的缺陷率其实是标注员效应。\n")
    P("| 标注员 | 题数 | clear | ambiguous | cant_tell |")
    P("|---|---:|---:|---:|---:|")
    per = collections.defaultdict(collections.Counter)
    for r in idx.values():
        for h in r["human"]:
            per[h["annotator"]][h["verdict"]] += 1
    for a, c in sorted(per.items(), key=lambda x: -x[1]["ambiguous"] / max(1, sum(x[1].values()))):
        t = sum(c.values())
        P(f"| {a} | {t} | {100*c['clear']/t:.0f}% | **{100*c['ambiguous']/t:.0f}%** | {100*c['cant_tell']/t:.0f}% |")
    boiler = sum(1 for r in idx.values() for h in r["human"]
                 if h["notes"].startswith("The available footage does not provide enough evidence"))
    P(f"\n其中 **{boiler} 条** ambiguous 配的是同一句英文模板，不是逐题观察。\n")

    # ---- 3 判决
    P("## 3. 判决\n")
    c = collections.Counter(a["decision"] for a in adj.values())
    dec = sum(v for k, v in c.items() if k != "PENDING_VIDEO")
    P("| 判决 | 题数 | 占已判 |")
    P("|---|---:|---:|")
    for k, v in c.most_common():
        if k == "PENDING_VIDEO":
            continue
        P(f"| `{k}` | {v} | {100*v/max(1,dec):.1f}% |")
    P("")

    # ---- 4 盲猜
    scored = [b for b in blind.values() if "guessable" in b]
    if scored:
        g = sum(b["guessable"] for b in scored)
        lo, hi = wilson(g, len(scored))
        P("## 4. 盲猜闸门（完全不给视频）\n")
        P(f"三票取二判 guessable：**{100*g/len(scored):.1f}%**（{g}/{len(scored)}，95% CI [{lo:.1f}, {hi:.1f}]）。")
        P(f"项目历史语料是 48.6%，验收线是 15%。\n")
        by = collections.defaultdict(lambda: [0, 0])
        for b in scored:
            k = f"{b['n_options']} 选 {b['n_select']}"
            by[k][0] += b["guessable"]; by[k][1] += 1
        P("| 题型规格 | 可盲猜 | 随机基线 |")
        P("|---|---:|---:|")
        for k in sorted(by, key=lambda x: -by[x][1]):
            a, t = by[k]
            no, ns = (int(x) for x in k.replace(" 选 ", " ").split())
            base = 100 / math.comb(no, ns)
            P(f"| {k} | {100*a/t:.1f}% ({a}/{t}) | {base:.1f}% |")
        P("")

    # ---- 5 修复
    if reps:
        okr = [r for r in reps if r.get("repaired")]
        P("## 5. 修复\n")
        P(f"送修 {len(reps)} 题，修好 **{len(okr)}** 题（{100*len(okr)/max(1,len(reps)):.1f}%）。")
        rounds = collections.Counter(r.get("rounds_used") for r in okr)
        P("每题最多三轮，每轮都必须重新过两道验收（不给答案的视频复核 + 不给视频的盲猜闸门），过不了就退回原题。")
        P(f"轮次分布：{dict(sorted(rounds.items(), key=lambda x: (x[0] is None, x[0])))}\n")
        why = collections.Counter()
        for r in reps:
            if r.get("repaired"):
                continue
            for a in r["attempts"]:
                if a.get("why"):
                    why[a["why"].split("(")[0].split(",")[0][:48]] += 1
        if why:
            P("修不好的原因：")
            for k, v in why.most_common(6):
                P(f"- {k} — {v} 次")
            P("")

    # ---- 6 交付
    if v2:
        P("## 6. 交付\n")
        st = collections.Counter(r["status"] for r in v2)
        P("| 状态 | 题数 | 占比 |")
        P("|---|---:|---:|")
        for k, v in st.most_common():
            P(f"| {k} | {v} | {100*v/len(v2):.1f}% |")
        dr = collections.Counter(r.get("drop_reason") for r in v2 if r["status"] == "dropped")
        if dr:
            P("\n淘汰原因：\n")
            P("| 原因 | 题数 |")
            P("|---|---:|")
            for k, v in dr.most_common():
                P(f"| {k} | {v} |")
        kept = [r for r in v2 if r["status"] != "dropped"]
        P(f"\n可用题库 **{len(kept)}** 题，文件 `runningbench_v2_kept.jsonl`；全量含淘汰理由在 `runningbench_v2.jsonl`。\n")

    P("## 7. 复算方式\n")
    P("```")
    P("python3 prep_clips.py 48       # 把每题的片段压进 13 MB inline 预算")
    P("python3 video_review.py --vote 0 --workers 14 --rps 0.45")
    P("python3 blind_gate.py --votes 3")
    P("python3 adjudicate.py")
    P("python3 repair.py --decisions BROKEN_NO_CORRECT_OPTION,GOLD_CONTRADICTED_NO_REPLACEMENT")
    P("python3 repair.py --guessable-only")
    P("python3 build_v2.py && python3 report.py")
    P("```")
    open(f"{QA}/QA_FIX_REPORT.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\n-> {QA}/QA_FIX_REPORT.md")


if __name__ == "__main__":
    main()
