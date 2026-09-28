# segments_60s 题型分类 + 视觉核验项目

**一句话现状**：`segments60s_kit` 的 6770 题（第一人称户外走路/跑步、60秒片段）已按 9 类打好标签，
排除 318 题不可答的之后，剩余 6452 题**全部**（100%，非抽样）对着真实画面做过 Gemini 视觉核验，
78.8% 标准答案通过核验。最终交付 **`final/segments_700_verified.jsonl`**：698 题，9 类全覆盖，全部通过核验。

**同步位置**：https://huggingface.co/datasets/taryya/RunningBench/tree/main/segments_60s_gemini_verification
（本地目录结构和 HF 上传路径一一对应，便于对照。）

---

## 1. 先看这三份文档，别的都是支撑材料

1. `docs/MY_CLASSIFICATION_SUMMARY.md` —— 9 类怎么分的、为什么这么分
2. `docs/VERIFICATION_SUMMARY.md` —— 核验怎么做的、精确通过率、抓到的真实错误案例
3. 本文件 —— 项目做到了第几步、文件都在哪、怎么复现

---

## 2. 目录结构

```
final/    ← 唯一权威数据产物, 只信这里
  segments_700_verified.jsonl   最终交付: 698题, 9类全覆盖, 全部 verify_overall=answer_correct
  segments_60s_typed.jsonl      6770题全量 + report_class(9类)/temporal_scope等字段
  verify_results.jsonl          6452题完整核验记录(逐选项 supported/ruled_out/undecidable + 整题判定)
  pass_rate_stats.json          9类精确的 池大小/通过数/通过率(全量普查值, 非抽样估计)
  excluded_short_clips.json     318题排除清单(视频片段过短或被截断, 数学上不可答)

docs/     ← 方法说明 + 仍在被引用的证据
  VERIFICATION_SUMMARY.md
  MY_CLASSIFICATION_SUMMARY.md
  question_type_spec.md          分类规范正文(判定优先级、被否决的方案)
  qa_taxonomy.json               项目方给出的官方5大类18小类定义, 供对照(见第5节, 尚未采用)
  question_types_relabeled_46.jsonl  46题人工gold标注, 是"分类一致率96%"这个数字的验证依据

code/     ← 现役流水线代码, 全部可复现
  caption_classify.py            9类分类脚本(基于caption字段派生, 零API调用, 秒级跑完全量)
  cut_clips.py                   视频切片(480p/crf30/去音轨)
  visual_verify.py               视觉核验(调Gemini看视频, 逐选项对账)
  floodgate.py                   API调用公共层(429退避重试, JSON兜底解析)
  build_final_verified_subset.py 从通过核验的题里分层抽样, 产出最终子集
  run_verify.sh / run_verify_round2.sh / run_verify_full.sh
                                  三轮核验各自的驱动脚本(切片+核验串联, 记录了实际怎么跑的)

  segments60s_kit_scripts/        官方kit自带的4个工具脚本(00~03, 9月15日), 见第3.5节
  blind_gate_repair_20260906/     blind_gate_source字段的生成代码(7个脚本, 9月6-7日), 见第3.5节
  runningbench_handoff_scripts/   cut_segment_recheck_clips.py——早于我这轮工作的同类尝试, 见第3.5节

archive/  ← 过程遗留, 保留是为了可追溯, 不是当前答案
  00_original_6_examples/        项目最初的6条示例标注, 引出了官方qa_taxonomy.json这条线索
  v1_v4_classification_iterations/  被caption分类取代的历史版本(v1~v4规则分类器、677题旧子集)
  abandoned_gemini_text_classification/  被caption分类取代的Gemini API分类尝试(只用文本, 未看视频)
  pipeline_intermediate_candidates/ 核验流水线的中间态候选文件(分轮次抽样的候选题列表)
```

---

## 3. 这个项目走过的路（为什么现在长这样）

**第一步：题型分类怎么定。** 最初按题干措辞写规则分类（v1~v4，见 `archive/v1_v4_classification_iterations/`），
也试过直接让 Gemini 读文本分类（见 `archive/abandoned_gemini_text_classification/`，一致率 96% 但要花 API 调用）。
后来发现 `segments60s_kit/README.md` 写明"题目就是照 caption 生成的"——于是改成**按 caption 的哪个字段派生来分类**
（`code/caption_classify.py`），零 API 调用、秒级跑完全量 6770 题，一致率同样 96%。这是现在唯一在用的分类方法。

**第二步：题目本身对不对，没人验证过。** `segments60s_kit/README.md` §6 写得很直白：这批题只过了盲猜闸门，
视觉核验和人工复核从未做过。所以光分类不够，还得对着画面核对每道题的标准答案。

**第三步：核验规模从抽样升级到全量普查。** 一开始只对 1551 题分层抽样核验（够用于估算各类通过率），
用户后来明确要求"对剩余约4900题也全部跑"，于是补齐到 **6452 题 100% 普查**。全量结果和抽样估计几乎完全吻合
（78.8% vs 78.7%），说明抽样当时就是准的，但现在是精确值。

**第四步：抓出一个抽样脚本的 bug。** 早期抽样脚本（`archive/v1_v4_classification_iterations/sample_subset.py`）
在"每视频题目上限"这个约束上有个 fallback 逻辑缺陷，配额不够时会悄悄解除上限。`code/build_final_verified_subset.py`
修正了这个问题，并对核验后的真实通过题池重新做了设计效应扫描，定出上限=15（缺口归零，设计效应1.28）。

## 3.5 补charge：`blind_gate_source` 字段的代码终于找到了

我在分类/难度分析里一直在用 `blind_gate_source` 这个字段（`original_never_guessable` 57.4%/`v2_repair` 22.9%/
`v1_repair` 19.7%），但之前手上一直没有产生这个字段的代码，只能从字段值反推含义。现在核实到了——
`code/blind_gate_repair_20260906/` 这 7 个脚本（9月6-7日，比 `qa_fix_gold_recovery_and_adjudication/` 那批
修复早了整整10天）就是源头，针对的是 `segments_60s` 这批 8,033 题（不是 `runningbench_v2` 那 1,487 题，两条线不重叠）：

- **`gate_segments_60s.py`**——盲猜闸门本身：不给视频，只给题干+选项，三票取二判 guessable。
  这一道闸门单独就淘汰了发布清单里 1,085 道拒绝题中的 542 道。
- **`pipeline.py`**——分阶段策略，每题在第一个能过关的阶段停下：原题盲猜→v1改写→v2对抗式改写(最多N轮)→
  仍不行就保留原题并标记 `still_guessable`。**关键设计**：v1 不会对本来就答不出的题下手——
  432 道单选题的实测中，无条件套用 v1 反而把其中 13 道原本没问题的题改坏了。
- **`repair_v2.py`**——为什么 v1 会卡在 73.4%：把干扰项改到"确实被画面否定"，天然会把干扰项改成
  "不常见的东西"，于是正确答案就成了选项里唯一"听起来正常"的那个，不用看视频靠常识就能选中。
  v1 的 prompt 完全没有要求干扰项的**先验可信度**要和正确答案打平，这是 v2 补的。
- **`repair_and_measure.py`**——修复效果怎么测: 同一批题**修复前**和**修复后**各盲猜一次(对照组+实验组同一次跑),
  不这样做的话,"闸门专挑猜中的题下手"这件事本身会造成回归均值假象,把噪声当成修复效果。
- **`repair_singles.py`**——只对单选题下手的理由: 587题配对实验测出改写让单选从95.4%降到73.4%可猜
  (McNemar p<0.0001),多选题完全没反应(51.6%→56.1%, p=0.48)。
- **`consolidate.py`**——四路措施汇总成一行一题, 优先级: 统一流水线判定 > v2修复成功 > v1修复成功 >
  原题从未被猜中 > 什么都没成功(标记不可用)。
- **`blind_test_videobench.py`**——`videobench`那1892题的官方"17.6%盲猜基线"是恒答上限, 不是实测,
  这个脚本补了一次真实的盲猜实测(分层抽样, 按题型和选项数拆开报告Wilson区间)。

`code/segments60s_kit_scripts/` 是官方 kit 自带的 4 个工具脚本（9月15日），这份说明文档里凡是提到
"480p 是下限"这类参数依据，源头都在 `02_cut_clips.py` 的注释里；`00_verify_kit.py` 是 kit 自检脚本，
`01_build_video_map.py` 就是这份项目一直在用的 `video_map.json` 的生成方式；`03_score.py` 是官方计分脚本，
含随机基线的正确算法（选项数3-8、答案数1-5共18种组合，随机基线不是1/n）。

**`code/runningbench_handoff_scripts/cut_segment_recheck_clips.py`——一次更早的、独立的同类尝试。**
这个脚本的docstring直接点破了`segments_60s`这批题的病根："这批题是照caption生成的，从没被对着画面复核过——
这正是盲猜模型能在上面拿48.6%的原因。要加视觉复核闸门就得把片段切回来。"这和我这轮工作的出发点完全一样，
但它当时只切出了276个缺失的片段（"原始片段除了276个都清理掉了"），说明视觉复核闸门在这条线上**没有真正跑起来**——
我这次做的6452题100%视觉核验，是这个缺口第一次被完整补上，不是重复劳动。参数上也互相印证：
两边都独立定出480p是下限、都在注释里提到"360p会把清晰路牌读成模糊、产生假的否定判定"这同一个教训。

---

## 4. 复现

```bash
cd segments_60s_gemini_verification

# 1. 分类(免费, 秒级; 需要能访问 segments60s_kit/captions/segments_60s.jsonl)
python3 code/caption_classify.py final/segments_60s_typed.jsonl /tmp/relabel.jsonl

# 2. 视觉核验(需要 Gemini/Floodgate API 访问; 6452题约6小时, 受API限流约束)
python3 code/cut_clips.py <候选文件.jsonl> <切片输出目录>
RB_WORKERS=6 python3 code/visual_verify.py <候选文件.jsonl> <切片输出目录> final/verify_results.jsonl

# 3. 重建最终子集(从 final/verify_results.jsonl 里 overall==answer_correct 的题分层抽样)
python3 code/build_final_verified_subset.py
```

---

## 5. 还没做完的事（如实列出，别当成"已完成"）

1. **和官方分类体系还没对上。** `docs/qa_taxonomy.json` 是项目方给的官方 5 大类 18 小类定义，
   和这里用的 9 类 `report_class` 是两套并行体系，从未做过映射。已知差异：本项目自建的
   `Text and OCR`、`Temporal Grounding` 不在官方 18 类里；官方的 `Counting QA`、`Negative or Absence QA`、
   按行人/车辆拆分的 Actor 类、整个 `Robustness QA` 大类，本项目都没有采用。
2. **核验错误没有被修正，只是被排除。** `final/verify_results.jsonl` 里判定为 `answer_wrong` 的 1364 题，
   标准答案字段原样保留在 `segments_60s_typed.jsonl` 里，没有被改写成 Gemini 认为对的选项——这是有意的保守选择
   （避免用一个模型的猜测覆盖另一个模型的错误，形成虚假的"已修正"数据）。如果要做这一步，见对话历史里讨论过的
   两条路径：半自动重写 vs 人工复核。
3. **核验只有单一模型**（gemini-3.1-pro-preview），没有多模型交叉验证或人工复核。

---

## 6. 和仓库里其它内容的关系

Hugging Face 仓库根目录的 `annotation_bundles/`、`JUDGING_MANUAL.md`、`校验操作手册.md`、以及各个
`annotation_<姓名>_<日期>.json` 针对的是**另一批 1526 题**（`fullvideo`/`excerpt`/`p01ma_gdrive`等来源的人工标注分发包），
和本项目处理的 `segments60s_kit` 6452 题是不同的题源，id 不重叠，两边互不影响。
