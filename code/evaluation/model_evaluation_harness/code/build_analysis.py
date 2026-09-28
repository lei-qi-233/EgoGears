#!/usr/bin/env python3
"""Compute per-model metrics + attributes + real statistics, dump one JSON for the report page."""
import json, math, os

EVAL = "/mnt/data/cvhci_video_understanding/eval"

ATTR = {
  "Qwen2.5-VL-7B":               dict(org="Alibaba",  total=7,   active=7,   moe=False, thinking=False, year=2025),
  "Qwen3-VL-8B":                 dict(org="Alibaba",  total=8,   active=8,   moe=False, thinking=False, year=2026),
  "Qwen3.5-9B":                  dict(org="Alibaba",  total=9,   active=9,   moe=False, thinking=False, year=2026),
  "Kimi-VL-A3B-Thinking-2506":   dict(org="Moonshot", total=16,  active=3,   moe=True,  thinking=True,  year=2025),
  "GLM-4.1V-9B-Thinking":        dict(org="Z.ai",     total=9,   active=9,   moe=False, thinking=True,  year=2025),
  "InternVL3.5-8B":              dict(org="OpenGVLab",total=8,   active=8,   moe=False, thinking=False, year=2025),
  "ERNIE-4.5-VL-28B-A3B":        dict(org="Baidu",    total=28,  active=3,   moe=True,  thinking=False, year=2025),
  "Qwen3-VL-30B-A3B-Thinking":   dict(org="Alibaba",  total=30,  active=3,   moe=True,  thinking=True,  year=2026),
  "Pixtral-12B-2409":            dict(org="Mistral",  total=12,  active=12,  moe=False, thinking=False, year=2024),
  "InternVL3.5-14B":             dict(org="OpenGVLab",total=14,  active=14,  moe=False, thinking=False, year=2025),
  "InternVL3.5-20B-A4B":         dict(org="OpenGVLab",total=20,  active=4,   moe=True,  thinking=False, year=2026),
  "Qwen3.5-27B":                 dict(org="Alibaba",  total=27,  active=27,  moe=False, thinking=False, year=2026),
  "InternVL3.5-30B-A3B":         dict(org="OpenGVLab",total=30,  active=3,   moe=True,  thinking=False, year=2025),
  "Qwen3.5-35B-A3B":             dict(org="Alibaba",  total=35,  active=3,   moe=True,  thinking=False, year=2026),
  "InternVL3.5-38B":             dict(org="OpenGVLab",total=38,  active=38,  moe=False, thinking=False, year=2025),
  "Qwen3.5-122B-A10B":           dict(org="Alibaba",  total=122, active=10,  moe=True,  thinking=False, year=2026),
  "InternVL3.5-241B-A28B":       dict(org="OpenGVLab",total=241, active=28,  moe=True,  thinking=False, year=2025),
  "Qwen3-VL-235B-A22B-Instruct": dict(org="Alibaba",  total=235, active=22,  moe=True,  thinking=False, year=2026),
  "Qwen3-VL-235B-A22B-Thinking": dict(org="Alibaba",  total=235, active=22,  moe=True,  thinking=True,  year=2026),
  "GLM-4.6V":                    dict(org="Z.ai",     total=106, active=None,moe=True,  thinking=False, year=2026),
  "Qwen2.5-VL-72B-Instruct":     dict(org="Alibaba",  total=72,  active=72,  moe=False, thinking=False, year=2025),
  "Gemma-3n-E4B-it":             dict(org="Google",   total=8,   active=4,   moe=False, thinking=False, year=2025),
  "Gemma-4-31B-it":              dict(org="Google",   total=31,  active=31,  moe=False, thinking=False, year=2026),
  "Gemma-4-26B-A4B-it":          dict(org="Google",   total=26,  active=4,   moe=True,  thinking=False, year=2026),
}

MIN_N = 700  # 排除样本太少的(Molmo2-8B n=26, Pixtral 部分分片, gemini 部分跑)


def load(tag):
    rows = []
    for l in open(f"{EVAL}/results/{tag}.jsonl"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if not r.get("error"):
            rows.append(r)
    return list({r["review_id"]: r for r in rows}.values())


def main():
    models = []
    for tag in sorted(ATTR):
        path = f"{EVAL}/results/{tag}.jsonl"
        if not os.path.exists(path):
            continue
        rows = load(tag)
        n = len(rows)
        if n < MIN_N:
            continue
        exact = sum(r["exact"] for r in rows)
        overlap = sum(r["overlap"] for r in rows) / n
        wellformed = sum(1 for r in rows if r["n_pred"] == r["n_select"]) / n
        p = exact / n
        se = (p * (1 - p) / n) ** 0.5
        by_unit = {}
        for r in rows:
            by_unit.setdefault(r["unit"], []).append(r["exact"])
        by_unit = {u: sum(v) / len(v) for u, v in by_unit.items()}
        a = ATTR[tag]
        models.append({
            "model": tag, "org": a["org"], "total_b": a["total"], "active_b": a["active"],
            "moe": a["moe"], "thinking": a["thinking"], "year": a["year"],
            "n": n, "exact": round(100 * p, 2), "exact_lo": round(100 * max(0, p - 1.96 * se), 2),
            "exact_hi": round(100 * min(1, p + 1.96 * se), 2),
            "overlap": round(100 * overlap, 2), "wellformed": round(100 * wellformed, 2),
            "by_unit": {u: round(100 * v, 1) for u, v in by_unit.items()},
        })

    # ---- 真实统计：log10(active_b) 与 exact 的 Pearson 相关，仅用 active_b 已知的模型 ----
    pts = [(math.log10(m["active_b"]), m["exact"]) for m in models if m["active_b"]]
    def pearson(pts):
        n = len(pts); mx = sum(x for x, _ in pts) / n; my = sum(y for _, y in pts) / n
        cov = sum((x - mx) * (y - my) for x, y in pts)
        sx = (sum((x - mx) ** 2 for x, _ in pts)) ** 0.5
        sy = (sum((y - my) ** 2 for _, y in pts)) ** 0.5
        return cov / (sx * sy) if sx and sy else 0.0
    r_active = pearson(pts)
    pts_total = [(math.log10(m["total_b"]), m["exact"]) for m in models]
    r_total = pearson(pts_total)

    moe_pts = [m["exact"] for m in models if m["moe"]]
    dense_pts = [m["exact"] for m in models if not m["moe"]]
    think_pts = [m["exact"] for m in models if m["thinking"]]
    nonthink_pts = [m["exact"] for m in models if not m["thinking"]]

    org_avg = {}
    for m in models:
        org_avg.setdefault(m["org"], []).append(m["exact"])
    org_avg = {o: round(sum(v) / len(v), 1) for o, v in sorted(org_avg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))}

    # 235B 同底座 Instruct vs Thinking 直接对照(唯一一对严格控制其它变量的样本)
    pair = {m["model"]: m["exact"] for m in models if "235B-A22B" in m["model"]}

    out = {
        "models": sorted(models, key=lambda m: -m["exact"]),
        "stats": {
            "n_models": len(models),
            "r_active_params": round(r_active, 3),
            "r_total_params": round(r_total, 3),
            "moe_mean": round(sum(moe_pts) / len(moe_pts), 1), "moe_n": len(moe_pts),
            "dense_mean": round(sum(dense_pts) / len(dense_pts), 1), "dense_n": len(dense_pts),
            "thinking_mean": round(sum(think_pts) / len(think_pts), 1), "thinking_n": len(think_pts),
            "nonthinking_mean": round(sum(nonthink_pts) / len(nonthink_pts), 1), "nonthinking_n": len(nonthink_pts),
            "org_avg": org_avg,
            "instruct_vs_thinking_235B": pair,
        },
    }
    json.dump(out, open(f"{EVAL}/analysis.json", "w"), ensure_ascii=False, indent=1)
    print(f"models={len(models)}  r(log active_params, exact)={r_active:.3f}  r(log total_params, exact)={r_total:.3f}")
    print(f"MoE mean={out['stats']['moe_mean']}% (n={out['stats']['moe_n']})  Dense mean={out['stats']['dense_mean']}% (n={out['stats']['dense_n']})")
    print(f"Thinking mean={out['stats']['thinking_mean']}% (n={out['stats']['thinking_n']})  Non-thinking mean={out['stats']['nonthinking_mean']}% (n={out['stats']['nonthinking_n']})")
    print("按厂商均分:", out['stats']['org_avg'])
    print("235B Instruct vs Thinking:", pair)


if __name__ == "__main__":
    main()
