# RunningBench 多视频/跨视频题标注流程

**这份文档不是我这次做的工作，是从项目内部存档（conductor：`s3://yuedong/cvhci_video_understanding/bundles/runningbench_handoff/`）
里找到、原样搬过来的现有方法论文档。** 我这次只做了汇总和整理，`source_docs/` 里是原始文件，未做任何删改。

**和 `segments_60s_gemini_verification/` 的关系**：那份是我自己做的工作（单视频 60 秒片段的分类+核验），
和这里讲的多视频/跨视频题是完全不同的题层，互不重叠，我的核验流水线**没有覆盖过**这里说的任何一层。

> **v2 修订**：v1 版本只讲了 `cross_video`（347题）一种，**低估了范围**。往下挖了 `qa_fix/master_index.jsonl`
> （1,526 题人工复核集的权威索引）才发现：真正"需要看不止一段录像"的题型一共有 **5 种**，不是 1 种，见第 2 节。

---

## 1. 全库五层的粗粒度划分

RunningBench 全库 11,802 道题按`CORPUS_OVERVIEW_README.md`的口径分五层：

| 层 | 题数 | 评测单元 |
|---|---:|---|
| `segments_60s` | 8,033 | 单个60秒片段 |
| `videobench` | 1,892 | 单个25秒片段 |
| `whole_video` | 1,411 | 完整单段录像 |
| `cross_video` | 347生成→135通过 | ≥2段录像 |
| `excerpts` | 119 | 1/2/4/7分钟摘录 |

**这张表的 `unit` 枚举是不完整的**——它只是全库去重统计用的粗口径，实际交付给人工复核的题，`unit` 字段还有另外三个
这张表里完全没出现的取值。往下看。

---

## 2. 更完整的账本：1,526 题人工复核集里真实的五分类

`source_docs/master_index_1526_reviewed.jsonl` 是这批题的权威索引，`unit` 字段实际有 5 个取值：

| unit | 题数 | 来源(source) | 需要几段录像 |
|---|---:|---|---|
| `whole_video` | 684 | `fullvideo`(615) + `excerpt`(69) | 1（但看的是整段/多个时间点的证据片段） |
| `multi_caption_multi_clip` | 305 | `p01ma_gdrive`(172) + `p01ma_hf`(133) | 1（同一段录像切出的多个时间片段，问顺序/变化） |
| **`anonymous_multi_clip`** | **273** | `rbma273`(273，一一对应) | **≥2——见下方实测证据** |
| `cross_video` | 135 | `fullvideo` | ≥2（题干直接写"VIDEO_A/VIDEO_B"） |
| `cross_recording` | 129 | `p01ma_gdrive`(92) + `p01ma_hf`(37) | ≥2（题干直接写"comparing … recordings"） |

**真正需要对照≥2段不同录像的三种合计 537 题**（`anonymous_multi_clip` 273 + `cross_video` 135 + `cross_recording` 129），
不是只有 `cross_video` 一种。`multi_caption_multi_clip`（305题）虽然也要看好几个视频文件，但它们是**同一段录像**切出来的
不同时间片段（`media_root`哈希相同，时间戳前后相接，如`CLIP_D_120_180`+`CLIP_E_420_480`），本质是"多片段"不是"多视频"。

### 2.1 `anonymous_multi_clip` 是怎么确认是跨录像的

题干只写"comparing … in CLIP_A, CLIP_C, and CLIP_H"，不像`cross_video`/`cross_recording`那样点明"两段录像"。
我抓了一道真实题（`RBMA-0114`，question_type=`turn-pattern-comparison`）的三个片段各截一帧对比：

```
CLIP_A: 清晰，正常步频
CLIP_C: 明显运动模糊，步频更快
CLIP_H: 严重拖影+光轨，接近奔跑
```

三帧里是**同一根路灯、同一片树**，但运动模糊程度依次递增——这正是项目"3速度(慢走/快走/跑)×2光照"设计里
**同一条路的三个速度版本**，只是这批题用的是`CLIP_*`字母标签而不是`VIDEO_*`。`rbma273`这个来源名本身
（273题精确对应）也印证这是独立于54段主录像之外的另一批专门做多速度对比的素材。

---

## 2.5 前传：最早的 Gemini 出题试点（8月24日，比 v1/v2 都早）

`FLOODGATE_ANNOTATION.md`（见 `qa_fix_gold_recovery_and_adjudication/docs/`）把
`scripts/gemini_video_pilot.py` 点名为"单次视频标注最小样例"。核实过时间戳——这三个脚本是
**全项目最早的 Gemini 视频标注代码**（8月24日），比 §4.2 的 v1（`generate_cross_video_qa.py`）、
v2（`build_fullvideo_annotations.py`）都早，是它们的原型，不是文档里点名缺失的
`repair_runningbench_annotations.py` 本身（那个文件确实没找回来，见 §6）。

三个脚本分工：

- **`gemini_video_pilot.py`**——单次调用的最小样例：切一段视频、送 Floodgate、要求 Gemini 返回结构化结果。
  文件开头写明"Floodgate 的 project token 在运行时从本机已配置好的流水线读取，绝不会拷进脚本或输出里"——
  这是后续所有脚本共用的凭证处理原则。
- **`run_gemini_annotation_pilot.py`**——批量编排：读一份"计划"(plan)，对每一条依次切片→调 Gemini→
  把结果收进可供人工复核的 JSONL。试点用的模型是 `gemini-3.6-flash`。
- **`apply_gemini_pilot_reviews.py`**——把人工对试点结果的复核意见合并回去，同时保留 Gemini 的原始提案
  （`ACTIVITY_MAP` 把 `FastWalk`/`SlowWalk`/`Running` 映射成标准活动标签），方便事后审计"模型最初提了什么、
  人改了什么"。

已收进 `source_docs/pilot_scripts_20260824/`。

## 3. 为什么能出这种题——数据采集设计

54 段录像，**3 参与者 × 3 速度 × 2 光照 × 3 轨迹**：

```
P01: 18段 · SlowWalk/FastWalk/Running × Day/Night · Traj01/02/03
P02: 18段 · 同上                                    · Trial01/02/03
P03: 18段 · 同上                                    · Traj01/02/03
```

**同一条轨迹有 6 个版本**（3速度×2光照）——同一条路在不同速度、不同光照下各走一遍，
这就是能问"这两段是不是同一条路"的物理基础。详见 `source_docs/录像清单.csv`（55条录像的完整元数据）。

> ⚠️ 元数据里的 `turnaround_sec`（折返时刻）和 `shape`（路线形状）**54 段全部是 medium/medium-low/low 置信度，
> 没有一段是 high**，P02 那 18 段的 shape 还是 `out_and_back_unconfirmed`（未确认）。这两个字段只能当参考，
> 标注者被要求自己看视频确认并记录下来。

---

## 4. 两条并行的生产线

### 4.1 人工标注流程（`source_docs/标注流程规范.md`，630行，v1，2026-08-31）

给拿到原始录像、负责人工出题的人看的操作手册。核心内容：

- **切片**：60秒片段（主力，从头到尾平铺切完不抽样）/ 180秒窗口（问顺序变化用）/ 任意时长组合（60秒片段拼接）/ 整段录像
- **Caption 五字段结构**：`overall_environment` / `objects_and_attributes` / `actions_and_events` / `trajectory_and_turnings` / `spatial_relations`
- **题目格式**：**一律8选项选3**，不做单选——因为单选6选项不看视频猜对率高达64.7%，8选项多选降到25.3%，配合严格干扰项规则能压到7.3%
- **跨录像题**专门列为一类问法："这两段录像是不是同一条路"
- **干扰项四条硬规则**（全流程最难的一步，48.6%的题不看视频能猜中全败在这一步）：
  1. 只用这个片段自己的东西造干扰项，不引入画面里没有的东西
  2. 靠"组合错"不靠"内容错"（顺序颠倒/左右互换/位置错配/幅度写错/主体调换）
  3. **不要让每个干扰项都只改一处**——曾经这么做过，盲猜率反而从48.6%升到75%（正确答案变成了"每栏众数"，逐栏投票就能还原）
  4. 八个选项长度和详细程度要接近（±25%以内），正确项不能是最长最具体的那个
- **自检四关**（顺序不可换）：打乱选项 → 盲猜测试(≥2/3命中即废题) → 480p视频复核 → 答案分布检查
- **12条已知踩坑清单**：文件名信息泄漏、正确项写太长、干扰项常识不可能、每处只改一样、排序题带时间戳、打乱顺序错了、复核分辨率不够、固定题数指标、代理指标代替不了盲猜实测、小样本下结论、直接信未确认的元数据、给1.3秒尾料出题

### 4.2 两版自动化脚本——`build_fullvideo_annotations.py` 是更新的那版

`source_docs/` 里其实有**两版**跨视频自动生成脚本，是同一条流水线的两个迭代：

**v1：`generate_cross_video_qa.py`**——只做跨视频题，针对一组`route_id`：切3分钟窗口 → `gemini-3.1-pro`逐窗口观察
(带时间戳的地标/事件/转弯,录像匿名A/B/C) → 合并成每段的绝对时间线 → `gemini-3.5-flash`写题(6种题型:
`route-phase-alignment`/`landmark-visibility-comparison`/`transient-event-attribution`/`entry-and-merge-comparison`/
`reverse-view-recognition`/`route-retrieval`) → 四道闸门 → 打乱选项。

**v2：`build_fullvideo_annotations.py`**——**同一个脚本里把`whole_video`和`cross_video`一起产出**：
1. **全局遍历**：整段录像360p/10fps(≤11MiB) → `gemini-3.1-pro`一次性看完整段(不切样本)，产出路线阶段/转弯/是否回到起点/
   重复经过的地标/环境阶段/动态事件
2. 已有的3分钟窗口观察(如果有)合并进来做`time_index`，只用于精确定位证据，不当采样
3. **`whole_video`题**：每段录像只用自己的标注出题(7种题型:`route-summary`/`start-end-relation`/`landmark-order`/
   `multi-step-turns`/`environment-stages`/`landmark-revisit`/`event-phase`——**这7个题型名和真实标注数据里的
   question_type分布完全对上**，确认这才是实际在用的版本)
4. **`cross_video`题**：同一个`route_id`下**所有录像的标注放在一起**出题(9种题型，`route-retrieval`等2种在v2里已废弃，
   见脚本内注释——"这两个视频是不是同一条路"这类"全局断言"用60秒窗口的证据无法坐实，复核时总能在证据之外的地方
   找到反例，v2换成了"两段在哪里分岔""哪对录像按顺序出现某组地标"这类可以被证据坐实的问法)
5. 五道闸门(见第5节)

**顺序值得记住**：`whole_video`先生成(每段录像自己的标注)，`cross_video`是**用多段`whole_video`标注拼出来的**——
两层共用同一次"全局遍历"的产出，不是两条独立流水线。

### 4.3 `04_脚本/` 目录里其余9个脚本——覆盖了每一层各自的短板

`generate_cross_video_qa.py`和`build_fullvideo_annotations.py`只是这个目录的一部分。补全后能看出
这条流水线其实是"哪层测出问题就专门写脚本修哪层"，不是一次性设计好的：

- **`blind_test_segments.py`**——测出`segments_60s`真实盲猜率48.6%(不是官方宣称的17.6%, 那个数字只是
  "选项字母分布均匀"，不是"题需要看视频")；**`repair_segment_distractors.py`**——照着测出来的问题重造干扰项，
  就是`segments_60s_gemini_verification/`里`blind_gate_repair_20260906/`那批脚本的同一条思路，但这里是
  更早的一版实现；**`export_segment_bundle.py`**——把修好的segment题打包成本地可跑闸门的自包含单元。
- **`annotate_missing_hf_60s.py`**——补7段HuggingFace录像缺失的60秒粒度标注，缺失不是随机的：
  同一场景的快/慢速版本都在，只有中间某个速度缺了60秒描述，像是标注批次中途断了没续上。
- **`annotate_windows_from_video.py`**——推翻了第一版180秒窗口题的做法：直接用已有60秒caption拼凑出的
  180秒标注写题，盲猜率68.4%，比它想改进的segment语料还差（复核本身没问题，3/76矛盾，纯粹是题目好猜）。
  改成让模型**真正看完整个180秒窗口**再写标注，才有了`build_window_questions.py`这层的题。
- **`build_window_questions.py`**——180秒窗口这层存在的理由：60秒只有一个阶段出不了转弯题，整段录像
  10-33分钟又太长，180秒正好落在"结构开始出现"的区间（摘录实验测过：产出率从1分钟42%涨到4分钟60%）。
- **`expand_mcq.py`**——纯文本扩充干扰项数量/每段题量，不解码不上传任何视频，读的是已有的`dense_annotations`。
- **`package_full_release.py`** / **`package_gdrive_release.py`**——两种打包范围: 全量(含HuggingFace+VideoBench,
  covers剩下9,925道未判定题里的5,324道) vs 仅Google Drive子集(只有这个来源才带`scene_id`/`route_id`/
  `turnaround_sec`这类路线元数据，路线类题目就是靠这个校验的)。

`cut_window_clips.py`（180秒窗口切片，见文件清单）也在这个目录，服务上面`build_window_questions.py`这条线。

### 4.4 补充的通用早期脚本

`pilot_scripts_20260824/`除了原来的4个Gemini试点脚本，还补了3个同期的通用基础设施：
`build_manifest.py`（给CVHCI录像集建元数据清单）、`download_gdrive_ranges.py`（Google Drive大文件下载遇到
限额页时按HTTP Range分片续传的兜底方案）、`make_contact_sheet.py`（从视频里等间隔抽帧拼联络表，人工核对用）。

---

## 5. 闸门是怎么做的（5步，顺序不可改）

```
① shuffle          题干SHA-256做种的确定性打乱
② structure        选项数/答案数/去重/证据时间段检查
③ leak_scan        黑名单扫描: 文件名碎片/拍摄速度词/"根据描述..."这类穿帮措辞
④ blind_guess ×3   不给视频答三次, ≥2次命中即淘汰(且跳过⑤, 省掉最贵的视频调用, 约省20%)
⑤ visual_recheck   按evidence_spans精确切480p片段, 问画面是否支持答案
```

**顺序死规则**：打乱必须在盲猜前（不打乱测会测出虚高数字，曾把56.6%误读成真实盲猜率——那是"正确答案爱放前面"的位置假象）；
盲猜必须在视觉复核前（省最贵的那步）；复核必须用480p（曾用360p把清晰路牌误判成"模糊"，废掉了本来正确的题）。

---

## 6. 当前语料状态（`screening_status` 五个取值）

| 状态 | 含义 | 题数 |
|---|---|---:|
| `usable` | 通过全部闸门，可直接用 | 792~793(两份文档口径略有差异,以`ANNOTATION_AUDIT_GUIDE.md`最新为准) |
| `ungated` | **从未判定，需要人工/自动筛选** | 9,925 |
| `rejected_blind_guessable` | 淘汰：不看视频就能答对 | 542 |
| `rejected_contradicted` | 淘汰：画面显示的与答案不一致 | 479 |
| `rejected_insufficient` | 淘汰：画面看不清，判不了 | 64 |

**`whole_video` + `cross_video` 这 1,758 道是唯一走完全部五道闸门的层**：

```
整视频  1,411 生成 → 658 通过 (42%)
跨视频    347 生成 → 135 通过 (39%)
```

**`segments_60s`（8,033道，就是我自己做核验的那批的来源）和 `videobench`（1,892道）在这份 conductor 文档写成时全部 `ungated`**——
这与我自己实测的结论一致（我这边对 6452 题池做的全量视觉核验，通过率78.8%，正是把这批"从未判定"的题变成"已判定"的工作）。

上面这五个状态是`cross_video`/`whole_video`这两层的口径。`anonymous_multi_clip`/`cross_recording`/
`multi_caption_multi_clip`这三层走的是另一套人工审核流程（`human`字段记录的是人工标注员逐选项对账的结果，
不是自动闸门），1,526题人工复核集里能看到已经有真人标注员（Di、Lei、Haiwen Sun等）在做这件事，
就是仓库根目录`annotation_bundles/`那10个包在收的东西。

---

## 7. 已知的坑，用之前必看

1. **`segments_60s_repaired_distractors.jsonl`（3,346道）不要计入总数**——这是同一批题的干扰项重造版本，
   没看视频、没跑闸门，851道（25%）一个新干扰项都没通过验证全部回退原文，等于没修。
2. **`videobench` 的 `blind_ceiling: 33.2` 是恒答上限，不是实测盲猜率**——同一错误在段级语料上差了近3倍
   （标称17.6% vs 实测48.6%）。videobench里444道"2选项"题随机基线就是50%，天然占优。
3. **原始计数29,129道，去重后11,802道**——六个非权威段级存档目录全是同一批题的旧版本，独有内容为0，
   按文件数而不是题干去重会让总数虚高2.5倍。
4. **`unit`字段的枚举本身就有两套口径**——`CORPUS_OVERVIEW_README.md`只讲`cross_video`一种多视频类型，
   `master_index.jsonl`（人工复核集的权威索引）讲了3种。**读文档判断"多视频题有多少"之前，先看的是哪份文档**——
   这正是v1版本这份说明文档少算了537-135=402道题的原因。

---

## 8. 文件清单

```
source_docs/
  标注流程规范.md                     人工标注操作手册(630行, 切片/caption/出题/干扰项/自检四关/12条踩坑)
  ANNOTATION_AUDIT_GUIDE.md           全流程审计手册(873行, 每层状态+每步用的模型和规则+人工该抽查哪里)
  CORPUS_OVERVIEW_README.md           全量语料总览(11,802题的层级构成/字段说明/闸门状态分布, unit枚举不完整见§1)
  generate_cross_video_qa.py          跨视频题自动生成脚本 v1(可运行, 含完整prompt)
  build_fullvideo_annotations.py      whole_video+cross_video统一生成脚本 v2(可运行, 是实际在用的版本, 见§4.2)
  master_index_1526_reviewed.jsonl    1,526题人工复核集的权威索引(真实5种unit的数据来源, 见§2)
  录像清单.csv                        55条录像的完整元数据(participant/speed/lighting/route_id/shape/turnaround_sec等)
  cut_window_clips.py                 切180秒窗口用的脚本(480p, 服务whole_video/cross_video的time_index证据)
  blind_test_segments.py              测出segments_60s真实盲猜率48.6%(见§4.3)
  repair_segment_distractors.py       照测出的问题重造segments_60s干扰项(见§4.3)
  export_segment_bundle.py            把修好的segment题打包成本地可跑闸门的自包含单元
  annotate_missing_hf_60s.py          补7段HuggingFace录像缺失的60秒粒度标注
  annotate_windows_from_video.py      让模型真正看完180秒窗口再写标注(取代盲猜率68.4%的第一版做法)
  build_window_questions.py           180秒窗口这层的出题脚本(见§4.3, 为什么是180秒)
  expand_mcq.py                       纯文本扩充干扰项数量/每段题量, 不碰视频
  package_full_release.py             全量打包(含HuggingFace+VideoBench, 给全体未判定题的人工筛选用)
  package_gdrive_release.py           仅Google Drive子集打包(唯一带route_id等路线元数据的来源)
  pilot_scripts_20260824/             全项目最早的Gemini视频标注试点(v1/v2的原型, 见§2.5)
    gemini_video_pilot.py               单次调用最小样例
    run_gemini_annotation_pilot.py      批量编排: 切片→调Gemini→组装成可复核JSONL
    apply_gemini_pilot_reviews.py       合并人工复核意见, 同时保留Gemini原始提案供审计
    make_video_clip.py                  更早的切片工具: 做隐私更安全、码率更低的试点用片段
    build_manifest.py                   给CVHCI录像集建元数据清单
    download_gdrive_ranges.py           Google Drive大文件下载遇限额时按HTTP Range续传的兜底方案
    make_contact_sheet.py               从视频等间隔抽帧拼联络表, 人工核对用
```

原始位置（conductor）：`s3://yuedong/cvhci_video_understanding/bundles/runningbench_handoff/`
（该前缀下还有 `03_文档/PREDICTION_QA_PLAN.md`、`03_文档/摘录跑批报告.md`、`02_全量数据/manifest.json` 等未搬运的相关文件，
`master_index_1526_reviewed.jsonl`原始位置是`qa_fix/master_index.jsonl`，需要更细节的可以从这两个前缀继续找。）
