# RunningBench 全量语料（Google Drive + HuggingFace）

**11,802 道选择题 + 1,795 条描述**，两个视频来源全部包含。
不含原始视频（22.1 小时，需另行拷贝）。

---

## 1. 一眼看清

```
11,802   生成过的选择题（题干去重后）
    792   ✅ 通过全部闸门，可直接评测        6.7%
  1,085   ❌ 闸门淘汰，带原因存档未删        9.2%
  9,925   ⬜ 从未判定 ← 人工筛选的对象     84.1%
```

**84% 的题没有被判定过。** 不等于不合格 —— 等于不知道。

| 来源 | 题数 |
|---|---|
| google_drive | 5,256 |
| huggingface | 6,546 |

---

## 2. 目录

```
questions/
  segments_60s.jsonl                       8033   60 秒片段题 ⬜ 全部未判定
  whole_video.jsonl                        1411   整视频题   ✅ 已过闸门
  cross_video.jsonl                         347   跨视频题   ✅ 已过闸门
  excerpts.jsonl                            119   摘录题     ✅ 已过闸门
  videobench.jsonl                         1892   25 秒片段题 ⬜ 全部未判定
  segments_60s_repaired_distractors.jsonl  3346   ⚠️ 见 §5，不计入总数

captions/
  segments_60s.jsonl                       1400   60 秒密集描述
  windows_180s.jsonl                         97   180 秒描述（看视频写的）
  windows_180s_aggregated.jsonl             130   ⚠️ 未看视频，由 60 秒聚合
  whole_video.jsonl                          19   整片描述
  whole_video_aggregated.jsonl               24   ⚠️ 未看视频
  fullvideo_structural.jsonl                125   整片结构标注（阶段/转弯/环境）

manifest.json
```

全部 JSONL，一行一条记录。

---

## 3. 题目字段

```json
{
  "id": "3f2a1b8c9d4e5f60",            题干哈希，全局唯一
  "layer": "segment_60s",              segment_60s | fullvideo | excerpt | videobench
  "unit": "60s_segment",               60s_segment | 25s_segment | whole_video | cross_video | excerpt
  "provenance": "huggingface",         google_drive | huggingface
  "video": "kit_scc_hochhaus_fast",    跨视频题这里是列表
  "video_ids": {"M": "...", "N": "..."},   仅跨视频题：标签→录像映射
  "span_sec": [120, 180],
  "question": "...",
  "question_type": "landmark-order",
  "options": {"A": "...", ...},
  "answer": ["B", "E", "G"],
  "arity": "multi",                    single | multi
  "n_options": 8,
  "option_order": "shuffled",
  "evidence_spans": [{"start_seconds": 12, "end_seconds": 20, "description": "..."}],
  "why_hard": "...",

  "screening_status": "ungated",       ← 最重要的字段，见下
  "gates": {                           走过闸门的才有，否则为 null
    "blind_guess_hits_of_3": 0,
    "visual_recheck": "supported",
    "recheck_notes": "画面在 0:12 显示..."
  }
}
```

### `screening_status` 的五个取值

| 值 | 含义 | 数量 |
|---|---|---|
| `usable` | 通过全部闸门，可直接用 | 792 |
| `ungated` | **从未判定，需要人工筛选** | 9,925 |
| `rejected_blind_guessable` | 淘汰：不看视频就能答对 | 542 |
| `rejected_contradicted` | 淘汰：画面显示的与答案不一致 | 479 |
| `rejected_insufficient` | 淘汰：画面看不清，判不了 | 64 |

**筛选工作就是把 9,925 道 `ungated` 变成其余四种之一。**
操作规范见 `ANNOTATION_AUDIT_GUIDE.md` 第 5 节。

---

## 4. 各层的状态与已知问题

### `segments_60s.jsonl` — 8,033 道 ⬜ 全部未判定

从 60 秒片段的**文字描述**出题，**从未看视频**，无任何闸门。

**抽样实测**（801 道，占 9.8%）：

| | 不看视频可答对 |
|---|---|
| 单选（4,672 道） | **64.7%** |
| 多选（3,361 道） | **25.3%** |
| 整体 | **48.6%**（95% CI [45.1, 52.0]） |

外推全库约 **3,900 道有问题，但不知道是哪些** —— 抽样只给比例，不给名单。

**回看片段已备好**：`/mnt/data/data_anno/runningbench_segment_repair/clips_480p/`
1,332 段 60 秒 480p，文件名对得上 `video` + `span_sec`。

### `whole_video.jsonl` / `cross_video.jsonl` — 1,758 道 ✅ 已过闸门

评测单元是**完整原始录像**，走过五道闸门（见 §6）。

```
整视频  1,411 生成 → 658 通过（42%）
跨视频    347 生成 → 135 通过（39%）
```

**跨视频题是这套语料唯一无可替代的部分** —— 需要同时对照多段录像，
问「这两段是不是同一条路」。有 14 道题要对照 8 段录像
（同一轨迹的 3 速度 × 2 光照版本）。

`video_ids` 是**标签到录像的映射**（`{"M": "P03_FastWalk...", "N": ...}`），
题干里用字母指代录像。

### `excerpts.jsonl` — 119 道 ✅ 已过闸门

为填平时长分布做的（1/2/4/7 分钟四档），全部来自 huggingface。

实测结论：**产出率随时长上升，但 1 分钟档明显吃亏**

| 时长 | 产出率 | 95% CI |
|---|---|---|
| 1 分钟 | 42% | [27, 58] |
| 2 分钟 | 53% | [37, 69] |
| 4 分钟 | 60% | [42, 75] |
| 7 分钟 | 60% | [42, 75] |

区间互相重叠，**不能声称「长多少个百分点」** —— 每档只 30 来道题。

### `videobench.jsonl` — 1,892 道 ⬜ 全部未判定

**全库最大的未知：从未做过盲猜实测。**

它的 `blind_ceiling: 33.2` 是**恒答上限**（「永远选 C」能得多少分），
不是实测盲猜率。段级语料在同一个错误上差了近 3 倍：

```
标称 17.6%（恒答上限）  vs  实测 48.6%（真盲猜）
```

**特别注意 444 道「2 选项」题** —— 二选一随机基线就是 50%，盲猜天然占优。

另有 numeric 1,017 道、yes_no 748 道**不在本包内**（无 options 字段），需另做统计。

---

## 5. ⚠️ `segments_60s_repaired_distractors.jsonl` — 3,346 道

**这不是新题，是同一批多选题的干扰项重造版本。不要计入总数。**

做了什么：用文字标注重造干扰项，规则改严（多处同改、正确项细节不得过半）。

**没做什么：**

- **没看视频** —— 输入是 `dense_annotations`，不是footage
- **没跑闸门** —— 用 `--no-blind` 跑的，效果未测
- **851 道（25%）一个新干扰项都没通过验证，全部回退原文**，等于没修
  （看 `fully_reverted` 字段）

整体干扰项重造率 **64%**，即三分之一的干扰项位置仍是旧的。

**用它之前必须先测。**

---

## 6. 闸门是怎么做的（那 1,877 道走过的流程）

```
① shuffle          确定性打乱（题干 SHA-256 做种）
② structure        选项数/答案数/去重/证据时间段
③ leak_scan        黑名单：文件名碎片、拍摄速度、「根据描述…」
④ blind_guess ×3   不给视频答三次，≥2 次命中 → 淘汰，且跳过 ⑤
⑤ visual_recheck   按 evidence_spans 精确切出 480p 片段，问画面是否支持答案
```

**顺序不可改：**

- **打乱必须在盲猜前** —— 生成器爱把答案放首位，盲猜模型也爱选首位；
  打乱前测会得到虚高读数（曾测出 56.6%，是位置假象）
- **盲猜必须在回看前** —— ④ 淘汰的题不走 ⑤，省掉最贵的视频调用（约省 20%）

**回看用 480p，比写标注那遍的 288p 更清晰** ——
曾用 360p，模型把清晰可读的路牌误判成「模糊」，废掉了正确的题。
**降分辨率会制造假阴性。**

---

## 7. 关于去重（读数字前请看）

盘上原始计数是 **29,129** 道，去重后 **11,802** 道。砍掉的 17,327 道全是副本：

| 非权威来源 | 题数 | 独有内容 |
|---|---|---|
| `conductor_release/annotations_60s` | 3,246 | **0** |
| `gdrive_videos_annotations` | 1,554 | **0** |
| `shaky_videos_annotations_full/*` | 3,246 | **0** |
| `shaky_videos_annotations/*` | 1,692 | **0** |
| `gap_repair/annotations_60s` | 828 | **0** |
| 重造版（替换干扰项） | 3,346 | **0** |

**六个段级目录全是同一批题的旧版本，一道独有的都没有。**
按文件数而不是题干去重，总数会虚高 2.5 倍。

---

## 8. 关于来源归属（一个易错点）

`Mikail` 和 `Markus` 是 **huggingface 的子目录**，不是独立来源。
它们的部分旧记录仍带 `/mnt/task_runtime/runningQA/shaky_videos/` 这个工作区路径。

**而 `shaky_videos_segmented/` 是所有来源共用的切片输出目录**，与来源无关：

```
/shaky_videos/Mikail/Asemwald_joggen.MOV               ← HF 原始（旧路径）
/shaky_videos_segmented/GDrive_Data/P01/...            ← gdrive 的切片输出
```

按「路径含 shaky_videos」判定会把 gdrive 录像标成 HF。
本包的 `provenance` 字段**只依据 `original_video_path`**，已修正。

---

## 9. 对外报数

**别说「11,802 道题」而不加限定。** 建议：

> **主张**：792 道逐题验证的题（整视频 658 + 跨视频 135）
> **补充**：另有 11,010 道，验证程度不同，详见审计文档

报总数时必须同时给出状态分布，否则会让人以为有 11,802 道可用题。

---

## 10. 配套文档

```
ANNOTATION_AUDIT_GUIDE.md   审计手册，§5 是人工筛选操作规范
标注流程规范.md              新出题的操作手册（切片/caption/出题/自检）
PREDICTION_QA_PLAN.md       预测题设计方案（未执行）
```
