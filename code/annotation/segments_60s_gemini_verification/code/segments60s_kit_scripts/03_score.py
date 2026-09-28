#!/usr/bin/env python3
"""给预测结果计分，并与正确算出的随机基线比较。

输入是一个 JSONL，每行 {"id": "<题目id>", "pred": ["A","C"]}。

两个指标：
  exact    严格全对：预测集合与答案集合完全相同才得分。部分正确不得分。
  overlap  选项级重合：|pred ∩ answer| / len(answer)。多选题上更能区分"全错"与"差一个"。

**随机基线不是 1/n。** 本子集选项数 3–8、答案数 1–5 都不统一，共 18 种组合，
必须按 C(n_options, len(answer)) 逐题算再加权。两条基线的加权方式还不一样：

  exact 基线   按**题数**加权          Σ c·1/C(n,k) / Σ c        = 11.05%
  overlap 基线 按**答案槽位**加权      Σ c·k²/n     / Σ c·k      = 36.1%

（两者均经 6,770 题蒙特卡洛验证：11.37% / 35.9%。按题数加权算 overlap 会得到
 错误的 28.4%。）

因此**只报一个总分是不可比的** —— 模型只要偏好回答单选题就能刷高总分
（6 选 1 随机基线 16.67%，8 选 3 只有 1.79%）。本脚本强制按组合拆分输出。

用法:
  python3 03_score.py --pred preds.jsonl
  python3 03_score.py --random    # 不给预测，直接打印随机基线
"""
import argparse, collections, json, math, random
from pathlib import Path


def baselines(rows):
    cnt = collections.Counter((r["n_options"], len(r["answer"])) for r in rows)
    ex = sum(c / math.comb(n, k) for (n, k), c in cnt.items()) / sum(cnt.values())
    num = sum(c * (k * k / n) for (n, k), c in cnt.items())
    den = sum(c * k for (n, k), c in cnt.items())
    return ex, num / den, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="JSONL: {id, pred}")
    ap.add_argument("--kit", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--random", action="store_true", help="只打印随机基线")
    a = ap.parse_args()

    rows = {r["id"]: r for r in
            (json.loads(l) for l in open(f"{a.kit}/questions/segments_60s.jsonl"))}
    bex, bov, cnt = baselines(list(rows.values()))
    print(f"题库 {len(rows)} 题")
    print(f"随机基线  exact {100*bex:.2f}%   overlap {100*bov:.1f}%\n")
    if a.random or not a.pred:
        print(f"{'n选k':>8s} {'题数':>6s} {'随机exact':>10s}")
        for (n, k), c in sorted(cnt.items()):
            print(f"{n:3d}选{k:<3d} {c:6d} {100/math.comb(n,k):9.2f}%")
        return

    preds = {}
    for l in open(a.pred):
        d = json.loads(l)
        p = d.get("pred")
        preds[d["id"]] = [p] if isinstance(p, str) else list(p or [])

    miss = set(rows) - set(preds)
    bad_letter = [i for i, p in preds.items()
                  if i in rows and not set(p) <= set(rows[i]["options"])]
    bad_count = [i for i, p in preds.items()
                 if i in rows and len(p) != len(rows[i]["answer"])]
    print(f"完整性检查：未作答 {len(miss)}；预测字母越界 {len(bad_letter)}；"
          f"选项个数不符 {len(bad_count)}")
    if bad_count:
        print("  ⚠️ 个数不符的题无法公平计分 —— 提问时必须告知要选几项（len(answer)）")
    print()

    def report(sel, label):
        sub = [(rows[i], preds[i]) for i in preds if i in rows and sel(rows[i])]
        if not sub: return
        ex = sum(sorted(p) == sorted(r["answer"]) for r, p in sub)
        ov = sum(len(set(p) & set(r["answer"])) for r, p in sub)
        tot = sum(len(r["answer"]) for r, p in sub)
        b_ex, b_ov, _ = baselines([r for r, _ in sub])
        print(f"{label:26s} n={len(sub):5d}  exact {100*ex/len(sub):5.1f}% "
              f"(基线 {100*b_ex:4.1f}%)  overlap {100*ov/tot:5.1f}% (基线 {100*b_ov:4.1f}%)")

    report(lambda r: True, "全部")
    print("\n按 (选项数, 答案数) —— 必看，否则总分不可比:")
    for (n, k) in sorted(cnt):
        report(lambda r, n=n, k=k: (r["n_options"], len(r["answer"])) == (n, k), f"  {n} 选 {k}")
    print("\n按来源:")
    for p in sorted({r["provenance"] for r in rows.values()}):
        report(lambda r, p=p: r["provenance"] == p, f"  {p}")
    print("\n按盲猜闸门来源（修复与否）:")
    for s in sorted({r.get("blind_gate_source") for r in rows.values()} - {None}):
        report(lambda r, s=s: r.get("blind_gate_source") == s, f"  {s}")


if __name__ == "__main__":
    main()
