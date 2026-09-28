#!/usr/bin/env python3
"""Publish the repaired corpus to the HF dataset, under v2/ and with the numbers recomputed
from the files being uploaded rather than copied from a report that may have moved on."""
import json, collections, os, sys
from huggingface_hub import HfApi

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
REPO = "taryya/RunningBench"
TOKEN = open("/mnt/tmp/claude-0/-mnt-task-runtime/c57d65fa-3a32-4466-b3ba-64c7feaf8c28/scratchpad/.hf_token").read().strip()

RELIABILITY = {"Yufan Chen": 1.00, "Di": 0.90, "Ruiping Liu/ Qian Yin": 0.80, "Chen Zhang": 0.765,
               "Haiwen Sun": 0.733, "Chengzhi Wu": 0.667, "zhihang chen": 0.60, "Xiaoye Wang": 0.75}


def readme():
    kept = [json.loads(l) for l in open(f"{QA}/runningbench_v2_kept.jsonl")]
    allq = [json.loads(l) for l in open(f"{QA}/runningbench_v2.jsonl")]
    st = collections.Counter(r["status"] for r in allq)
    dr = collections.Counter(r["drop_reason"] for r in allq if r["status"] == "dropped")
    un = collections.Counter(r["unit"] for r in kept)
    qt = collections.Counter(r["question_type"] for r in kept)
    hd = [r for r in kept if r.get("human_decided_by")]
    hdc = collections.Counter(r["human_decided_by"] for r in hd)
    exp = sum(c * RELIABILITY[a] for a, c in hdc.items())
    multi = sum(1 for r in kept if len(r["clips"]) > 1)
    arity = collections.Counter(r.get("n_select") for r in kept)
    nopt = collections.Counter(len(r["options"]) for r in kept)

    L = []
    w = L.append
    w("# RunningBench v2 — 复核与修复后的题库\n")
    w(f"**{len(kept)} 题**，来自 `annotation_bundles/` 的 {len(allq)} 题。每一题都用真实视频逐选项复核过，")
    w("答案有问题的题要么改键、要么带视频重写选项、要么淘汰。\n")
    w("> ⚠️ **本目录含标准答案。** `annotation_bundles/` 的题包是刻意不含答案的，标注员在不知道答案的前提下盲判。")
    w("> 只要还有标注员在做题，就不要把本目录的访问权给他们。\n")
    w("## 交付构成\n")
    w("| 状态 | 题数 | 含义 |")
    w("|---|---:|---|")
    w(f"| confirmed | {st['confirmed']} | 两个互不知情的观察者与原答案一致 |")
    w(f"| repaired | {st['repaired']} | 选项经带视频重写，并重新通过视频复核 + 盲猜闸门 |")
    w(f"| rekeyed | {st['rekeyed']} | 答案键被改到证据支持的选项 |")
    w(f"| dropped | {st['dropped']} | 淘汰，不在 `runningbench_v2_kept.jsonl` 里 |\n")
    w("淘汰原因：\n")
    w("| 原因 | 题数 |")
    w("|---|---:|")
    for k, v in dr.most_common():
        w(f"| `{k}` | {v} |")
    w("\n## 文件\n")
    w("| 文件 | 内容 |")
    w("|---|---|")
    w(f"| `runningbench_v2_kept.jsonl` | 可用题库，{len(kept)} 行 |")
    w(f"| `runningbench_v2.jsonl` | 全量 {len(allq)} 行，含淘汰题与淘汰理由 |")
    w("| `QA_FIX_REPORT.md` | 复核与修复的完整报告 |\n")
    w("## 一条记录\n")
    w("```jsonc")
    w('{')
    w('  "review_id": "…",           // 与题包一致')
    w('  "unit": "cross_video",       // 题的单位，见下表')
    w('  "question": "…", "options": {"A": "…"}, "n_select": 3,')
    w('  "answer": ["A","C","F"],     // 恒为列表，单选也是长度 1')
    w('  "clips": [{"label":"VIDEO_A","path":"media/<review_id>/VIDEO_A_0_66.mp4"}],')
    w('  "status": "repaired",        // confirmed | repaired | rekeyed')
    w('  "evidence": …,               // 支持该答案的证据')
    w('  "evidence_source": "human",  // human 或 gemini_v0，仅 rekeyed 有')
    w('  "human_decided_by": "…",     // 若答案由单个标注员定夺')
    w('  "settled_by_crossmodel": true // 若由第三个模型打破僵局')
    w('}')
    w("```\n")
    w("`clips[].path` 相对各自 `annotation_bundles/bundle_XX.tar` 解包后的根目录。\n")
    w("## 分布\n")
    w("| 单位 | 题数 |")
    w("|---|---:|")
    for k, v in un.most_common():
        w(f"| `{k}` | {v} |")
    w(f"\n多片段题 {multi} / {len(kept)}。答案个数 " +
      "、".join(f"{k} 项 {v}" for k, v in sorted(arity.items())) + "；选项数 " +
      "、".join(f"{k} 选 {v}" for k, v in sorted(nopt.items())) + "。\n")
    w("题型：\n")
    w("| 题型 | 题数 |")
    w("|---|---:|")
    for k, v in qt.most_common():
        w(f"| `{k}` | {v} |")
    w("\n## 怎么做出来的\n")
    w("每题都把真实片段送给模型，逐选项判 supported / ruled_out / undecidable 并给出 `CLIP_x MM:SS`，")
    w("全程不出示答案；再与看了视频的人工标注、以及恢复出的原答案三方交叉。")
    w("答案被画面否定的题进入带视频的重写，重写结果必须重新通过两道验收（不给答案的视频复核 + 不给视频的盲猜闸门），")
    w("过不了就退回原题并如实标记，**不接受「尝试过」当作证据**。\n")
    w("分工用了不同模型，避免自证：\n")
    w("| 环节 | 模型 |")
    w("|---|---|")
    w("| 视频复核（验收） | `gemini-3.1-pro-preview` |")
    w("| 选项重写（写手） | `gemini-3.8-flash` |")
    w("| 盲猜闸门 | `gemini-3.5-flash` |")
    w("| 僵局第三票 | `gemini-3.8-flash` |\n")
    w("## 已知限制（用之前请读）\n")
    w("1. **三个模型同属 Gemini 家族。** 换代确实带来独立信息——在 60 道「人工与原答案一致而 3.1-pro 答错」的题上，")
    w("   3.8-flash 命中 41.7%（其 verdict 为 clear 时 49.0%），对比加权随机基线 9.0%，p=1.6e-11。")
    w("   但这不等于跨家族验证。真正的独立验收仍需非 Gemini 模型或人工抽检。")
    w(f"2. **{len(hd)} 题的答案由单个标注员定夺**（字段 `human_decided_by`）。按各人实测准确率加权，")
    w(f"   预期正确约 {exp/max(1,len(hd))*100:.0f}%。要更保守就按该字段过滤。各人可靠度在高置信集（n=124，随机基线 8.9%）上实测为：")
    w("   " + "、".join(f"{a} {RELIABILITY[a]*100:.0f}%" for a in sorted(RELIABILITY, key=lambda x: -RELIABILITY[x])) + "。")
    w("3. **两个来源必须当作独立的域**，不能合并后随机切分：同一路线被不同参与者、速度、光照反复走过，")
    w("   随机切分会造成严重场景泄漏。最低按 `video` 分组，推荐按 `(participant, route_index)`。")
    w("4. **音频是完美的参与者捷径**（P01 有、P02 无、P03 有），做任何评测都应去掉音轨。\n")
    return "\n".join(L) + "\n"


def main():
    api = HfApi(token=TOKEN)
    path = f"{QA}/_v2_README.md"
    open(path, "w").write(readme())
    files = [(f"{QA}/runningbench_v2_kept.jsonl", "v2/runningbench_v2_kept.jsonl"),
             (f"{QA}/runningbench_v2.jsonl", "v2/runningbench_v2.jsonl"),
             (f"{QA}/QA_FIX_REPORT.md", "v2/QA_FIX_REPORT.md"),
             (path, "v2/README.md")]
    for src, dst in files:
        print(f"  {dst:40s} {os.path.getsize(src)/1024:8.1f} KB")
    if "--dry-run" in sys.argv:
        print("\n--- dry run, 未上传 ---")
        print(open(path).read())
        return
    for src, dst in files:
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=REPO,
                        repo_type="dataset", commit_message=f"v2 corpus: {os.path.basename(dst)}")
        print(f"  ✓ {dst}")


if __name__ == "__main__":
    main()
