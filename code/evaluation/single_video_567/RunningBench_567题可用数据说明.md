# RunningBench 567 道可用题：数据说明

## 概览

本目录整理出一份可用于后续数据处理、模型评测或人工复核的 RunningBench QA 子集，共 **567 道题**。该子集从现有 698 道 QA 中筛选，所有题目均保留 `question_id`、视频标识、时间区间、题干、选项和答案。

| 来源 | 数量 | 纳入条件 |
|---|---:|---|
| 人工完整且结论明确 | 562 | 标注记录 `complete=true`，人工 verdict 为 `clear`，且有最终答案 |
| 不完整标注经补救 | 5 | 通过视频核验后恢复原答案，或修订题干/选项及答案后可用 |
| **合计** | **567** | 两类 question_id 不重叠，已去重 |

按答案选项数量统计，包含 **325 道单选题**和 **242 道多选题**。

## 文件

- `runningbench_567_human_usable.jsonl`：正式 567 题数据，每行一个 JSON 对象。
- `runningbench_567_inclusion_ledger.csv`：逐题纳入台账，记录来源类别、补救状态和答案。
- `runningbench_qa_human_corrected.jsonl`：完整的 698 题可追溯修订版；567 题子集从此文件筛出。
- `README_修订说明.md`：整体人工答案筛选和修订统计。
- `incomplete_annotation_repairs.csv`：不完整人工标注题的逐题补救记录及证据。
- `video_case_checks.csv`：视频抽查的逐题结论和证据摘要。

## JSONL 字段

每一行是一道题，保留原 QA 的字段，主要包括：

- `question_id`：题目唯一标识。
- `category`：题目类别。
- `segment`：视频片段描述及时间范围。
- `video_id`：对应视频标识。
- `start_second`、`end_second`：片段起止秒数。
- `question`：题干。
- `options`：选项对象，键通常为 A、B、C 等。
- `answer`：正确选项数组；单选题数组长度为 1，多选题大于 1。
- `usability_source`：本清单纳入来源，取值为 `complete_clear_human_annotation` 或 `repaired_incomplete_annotation`。
- `repair_status`：补救题对应的修复状态；原本完整清晰的人工题为空字符串。

## 筛选与修订口径

1. 对完整清晰的人工记录，采用人工最终答案。全部 562 道满足清晰条件的题均保留；视频抽查仅覆盖部分题，不应将整份 567 题清单描述为“全部逐题视频验证”。
2. 对不完整记录，只有视频证据足以确认答案，或能通过收窄题干、移除无法证实的选项并重设答案而修复的题，才补入清单。
3. 不确定、无法判断、视频证据不足，或视频中没有正确选项的题，不纳入这 567 道。
4. 原始 QA 和人工标注导出文件没有被覆盖；本数据是可追溯的衍生清单。

## 使用建议

- 训练或评测时用 `question_id` 作为稳定主键。
- 若只需要单选题，筛选 `answer` 数组长度为 1（当前 325 道）。
- 若将补救题与常规人工题分开分析，可按 `usability_source` 筛选，并结合 `repair_status` 查阅补救记录。
- 发布或共享时保留本说明、纳入台账和修订记录，避免丢失答案来源与修订依据。

## 可复现性

筛选逻辑脚本为 `work/make_usable_567.ps1`。在包含 `outputs/human-answer-repair` 数据文件以及 `%TEMP%/rb-human-review` 人工导出和受保护 QA 源文件的工作环境中运行：

```powershell
./work/make_usable_567.ps1
```

脚本会重新生成 567 题 JSONL 和纳入台账，并检查总数及题目 ID 是否重复。
