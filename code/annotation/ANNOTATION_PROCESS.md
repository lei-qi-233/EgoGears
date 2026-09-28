# 标注与核验流程详解 — 单视频题 vs 多视频题

本文档详细说明 RunningBench 两套语料里的题目是怎么生成、怎么标注、怎么核验的，
核心是把**单视频（单片段）题**和**多视频（多片段）题**的流程差异讲清楚——这两类题
在出题规则、标注步骤、质量闸门上都不一样，混着看容易误解数据的可信度。

## 0. 一共有三套独立的核验工作流

| 工作流 | 面向语料 | 是否人工 | 是否盲答 |
|---|---|---|---|
| **A. HF 题包众包复核**（10 包共 1526 题，实际回收 1243 条判定） | 语料一混合题源 | 是，10 位标注员 | **不做**——直接看视频逐选项核对 |
| **B. CLI 手工校验包**（1285 题） | 同上 | 是 | **强制先盲答**，程序锁死顺序，看完视频才能改 |
| **C. segments_700 视觉核验**（6452 题全量普查→698题） | 语料二（纯单片段） | **否**，全程 `gemini-3.1-pro-preview` | 不适用 |

A 和 B 用的是同一份《校验操作手册》，A 只是在开头声明"忽略盲答相关段落"。
下文的标注步骤，除非特别说明，指的是 A/B 共用的这份手册。

## 1. 三套片段标签，一眼看出这题是单视频还是多视频

看题目 JSON 里 `clips[].label` 就能立刻分辨：

| 标签形式 | 含义 | 出现在哪些 `unit` |
|---|---|---|
| `CLIP_1, CLIP_2 …` | **数字=真实时间顺序**（实测479道多clip题里标签序=时间序，无一例外） | `whole_video`（同一段录像内部） |
| `CLIP_A … CLIP_J` | **字母=随机发牌的展示序，刻意打乱、不等于时间序**（实测167/304、74/129题字母序≠时间序） | `multi_caption_multi_clip`、`cross_recording`、`anonymous_multi_clip` |
| `VIDEO_A … VIDEO_R` | **字母=一段独立录像的匿名代号**，不是时间；同一录像的多个证据窗口共用同一字母 | `cross_video`（跨不同录像） |

字母乱序是代码里**故意做的**，不是疏漏：

```python
# plan_bundles.py
labels = [chr(ord("A") + i) for i in range(len(chosen))]
order = list(range(len(chosen)))
rng.shuffle(order)   # presentation order must not reveal chronology
```

真实时间顺序被藏进一个只给出题模型用来"钉答案"、绝不出现在题面里的私有字段
（`private_gold_order_for_annotation_only`）。**跨录像题连这个私有字段都不给**——
因为两段录像没有共同的时钟，给了反而是泄题。这条设计原则写在流水线代码开头：

> Clip labels are dealt in a random presentation order, so CLIP_A is not the earliest
> segment. **A question whose answer can be read off the label sequence tests nothing.**

## 2. 单视频题的标注流程（简单版）

适用范围：`whole_video` 里只有 1 个 clip 的 173 道题，以及**语料二 segments_700 全部
698 题**（清一色 60 秒单片段）。

### 2.1 语料一里的单片段题：走人工四步法

```
① 看：完整看完这一个片段，边看边记一条时间线
   （转向方向+角度、路面变化、带数字/文字/颜色的物体、行人车辆、显著建筑，±3秒即可）
② 逐选项核对：一个一个过，不凭整体印象
   ✓ 支持（画面里找得到）  ✗ 排除（画面明确否定，一处细节错即为错）  ? 无法判定（画面没拍到）
③ 最终答案：勾恰好 n 项，✓ 数量对不上时不要硬凑，判定改选 ambiguous
④ 判定 + 备注：clear / ambiguous / cant_tell，写清楚哪个选项有问题
```

单片段没有"跨片段推理"这一步，是最简单的情况。**工时约 2-3 分钟/题**。

### 2.2 语料二 segments_700：全程模型核验，没有人工环节

流程：切片（480p / crf30 / 去音轨）→ 送 `gemini-3.1-pro-preview` 逐选项核对 →
输出 `supported`/`ruled_out`/`undecidable` 三态判定，以及整题的
`answer_correct`/`answer_wrong`/`insufficient`。

**规模：6452 次视频核验调用，100% 全量普查，不是抽样**，耗时约 6 小时。

一条关键方法论约束（避免循环论证）：

> **核验必须对着画面做，不能用 caption 核**：题目就是照 caption 生成的，caption 自身
> 的错误会原样传给题目，拿它核检不出真错误。

结果：`answer_correct` 5081 / `answer_wrong` 1364 / `insufficient` 7，
**整体通过率 78.8%**，最终按 9 类题型分层抽样出 698 题。

局限（原文档如实写明）：**单一模型、无多模型交叉验证、无人工复核，核验自身的错误率
未知**——这是语料二相比语料一的一个真实短板，语料一好歹有真人标注员做二次确认。

## 3. 多视频题的标注流程（比单视频多做四件事）

适用范围：`cross_video`（132题，跨不同录像）、`cross_recording`（129题，同路线不同
录像对比）、`multi_caption_multi_clip`（304题）、`anonymous_multi_clip`（270题）——
合计 835 题，占语料一（1487题）的 56.2%。

### 3.1 每个片段各记一条独立时间线（而不是一条）

> 多片段的题:**每个片段各记一条时间线**,标签写在开头。**标签顺序不代表时间顺序**,
> CLIP_A 未必比 CLIP_B 早。

### 3.2 额外一步：自己把片段之间的先后顺序推出来

这是单片段题完全不存在的步骤：

> 还差一件事:**CLIP_C 和 CLIP_F 谁先谁后?** 标签不代表顺序。……你得从画面推
> （比如 CLIP_C 结束处的场景是否衔接 CLIP_F 开头,或看两段的疲劳程度/环境延续性）,
> **推不出就依靠单段内部能判定的选项**。

### 3.3 逐选项对账时要先分类"段内可判"还是"跨段可判"

真题示范（节选）：

```
A  骑车人(C 0:15) 早于建筑11(C 0:29)      → ✓ 同段内可判
B  护柱(F 0:05) 早于围栏(F 0:47)          → ✓ 同段内可判
H  车架(C 0:41,鹅卵石) 早于草坪直路(F)    → ✓ 需要跨段顺序,由画面衔接判定
```

### 3.4 跨录像题的三条额外规则

> - 光照可以完全不同（一次白天一次夜里）,**别因为"看起来不像"就判不同地点**
> - 判"同一地点"要靠**不变的东西**:建筑形状、路的走向、固定设施
> - 判"不同"要靠**结构差异**:转弯序列不同、路面材质不同,而不是行人车辆这些偶然物

### 3.5 折返路段的左右翻转

> **去程右侧 = 回程左侧**,别被折返骗了

常见错误清单里排第一的两条都是多视频题特有的：折返忘了左右互换（spatial类题全判反）、
把 CLIP 标签顺序当时间顺序（event-order题判反）、跨录像因光照判错地点（昼夜对比题全错）。

### 3.6 工时是单片段的两倍

> 单片段题约 2-3 分钟/道,**多片段对比题 4-6 分钟/道**。一天 6 小时约 80-100 道。

## 4. 自动出题流水线：多片段题走的闸门比单视频题更严

| 题源 | 题数 | 生成方式 | 闸门数 |
|---|---:|---|---:|
| `fullvideo` | 741 | 模型看完整录像（10-20分钟）写结构标注→出题 | **5道** |
| `excerpt` | 66 | 1-7分钟摘录出题 | 5道 |
| `p01ma_gdrive`/`p01ma_hf` | 434 | 模型看同一录像的2-3个60秒片段出对比题 | **8道** |

### 4.1 单视频线的五道闸门

```
① 确定性洗牌(按题干哈希做种子)  ② 结构检查  ③ 泄漏黑名单
④ 盲猜×3(≥2/3命中就淘汰,不进⑤)  ⑤ 高清视觉回看(480p精确切在证据处)
```

顺序不可颠倒的理由：「洗牌前测试出过 56.6% 的虚高盲猜率，那是位置线索不是真泄漏」；
「盲猜没过的题直接跳过⑤，省下流水线里最贵的一步调用（约20%的题）」。

### 4.2 多片段线的八道闸门（实际更严格）

```
① 意图闸(题型在允许集合内,片段数量够)  ② 结构闸(8选项,3个正确项不能是最长的三项,
   选项长度须在中位数0.65-1.35倍内)  ③ 泄漏黑名单(不许出现trial/文件名/"根据描述"等)
④ 确定性洗牌  ⑤ 跨批去重(4层:哈希全等/词汇近重复/证据类型复用/签名)
⑥ 盲猜×3(不合格进重写循环,最多2轮,不是直接废题)  ⑦ 证据切片(片段切不出来就淘汰)
⑧ 逐选项视觉复核(每次最多带3个片段,不给caption不给答案,supported必须恰好等于标注答案)
```

多片段线独有两处机制：**出题时模型也要看片段**（不是只读caption），原因是
"只读描述出的选项，复核器在大多数题里只能确认三个正确项里的两个"；以及
**喂给出题模型的caption会被洗掉所有字面时间戳**，只留内容和顺序，让"谁先谁后"
没法从文字直接读出来。

### 4.3 实测：闸门通过率，多片段/跨视频更容易被刷掉

| 层 | 总数 | 通过 | 盲猜淘汰 | 视觉回看判矛盾 | 证据不足 |
|---|---:|---:|---:|---:|---:|
| whole_video | 1411 | 592 | 407 | 377 | 35 |
| cross_video | 347 | 135 | 103 | 81 | 28 |
| excerpts | 119 | 65 | 32 | 21 | 1 |
| **合计** | **1877** | **792 (42.2%)** | 542 | 479 | 64 |

**盲猜活下来的1335题里，视觉回看只过了792题（59.3%）**——即便题目不能被盲猜，
仍有约40%的标注答案与画面对不上，靠视觉回看这道闸门刷掉了。

实测盲猜率对比也印证了"多片段更抗盲猜"这个设计目标确实达到了：

| 语料 | 盲猜可猜率 |
|---|---|
| 旧的无闸门单片段题 | 76% |
| 整片线8选项多选题 | **7.3%** |
| P01MA新题(含多片段) | 30% |

跨录像题（`route_cross_condition`）存在的原因，代码注释写得很直白：

> 只给单一录像素材时，模型会挑画面里最显眼的东西写选项，盲猜率升到29.9%。
> 对比题没有这种捷径。

## 5. 跨录像题的三道防泄漏措施（代码级）

1. **模型从不被告知哪些片段来自同一次录制**——这正是"是不是同一地点"这道题要问的东西
2. **展示顺序不按录像聚类**（实测：0/30 例泄漏）
3. **跨录像题不给私有时间顺序**（两段录像没有共同时钟，没法给）

对应发给出题模型的原始提示语：

```python
note = ("一些片段来自沿同一路线走的另一次录制。哪些片段属于同一次录制未知，
        且没有任何片段与其它片段是连续的。" if cross else
        "全部来自同一次连续录制。")
```

## 6. 单/多视频各自的题型分类（完整表）

### 6.1 语料一（1487题，`question_type` 完整29类，按 unit 严格分区)

| unit | 专属题型（每类最少所需片段数≥2的都是多片段题型） |
|---|---|
| `whole_video`（652题，单片段为主） | start-end-relation · event-phase · landmark-order · environment-stages · route-summary · multi-step-turns · landmark-revisit（共7类） |
| `cross_video`（132题） | unique-transient-obstacle · landmark-visibility-comparison · route-phase-alignment · description-matching · reverse-view-recognition · different-start-shared-path · route-divergence-location · divergence-point · landmark-order-discrimination · route-retrieval · same-route-identification（共11类） |
| `multi_caption_multi_clip`+`anonymous_multi_clip`（304+270题） | event-order · turn-pattern-comparison · route-phase-discrimination · spatial-consistency · landmark-cooccurrence · environment-transition · cross-segment-revisit（共7类，每类最少2-3片段） |
| `cross_recording`（129题） | same-place-different-recording · shared-landmark-identification · route-identity-discrimination · environment-difference-across-recordings（共4类） |

按能力归并的6大维度、题数占比：Temporal sequence and phase（516/34.7%）·
Ego-motion and route geometry（406/27.3%）· Landmark memory and occurrence（206/13.9%）·
Spatial relation and viewpoint（154/10.4%）· Environmental state and transition（113/7.6%）·
Place/route matching and retrieval（92/6.2%）。

**曾经被砍掉的两类多片段题型**是个值得记录的设计教训：`same-route-identification`
（9题只有1题可用）和 `route-retrieval`（20题只有2题可用）——它们问的是整片级的
全称命题（"是不是同一条路线"），任何≤60秒的证据窗口都判不了，回看器总能在窗口
之外找到反例。后来换成了"两段视频在哪里分岔"这种答案能从窗口本身判定的问法。

### 6.2 语料二（698题，纯单片段，9类 `report_class`）

| 类别 | 6452题池通过率 | 最终698题占比 |
|---|---:|---:|
| Object and Attribute QA | 83.4% | 35.8% |
| Ego-relative Spatial Relation | 76.6% | 27.9% |
| Trajectory-grounded QA | 70.8% | 15.2% |
| Ego-motion Recognition | 79.8% | 6.3% |
| Temporal Grounding | 76.8% | 3.0% |
| Scene and Place QA | 86.5% | 3.0% |
| Event Sequencing | 74.6% | 3.0% |
| Actor Action and Motion | 80.0% | 3.0% |
| Text and OCR | 86.4% | 2.7% |

通过率最低的两类恰好都跟"顺序/方向"有关（Trajectory 70.8%、Event Sequencing
74.6%），跟错误案例里最常见的两种问题（左右反转、事件顺序错）对得上。

分类依据是"正确项与干扰项的区分点，不是题干措辞"——题干是模板生成的，措辞本身
不携带标注信息。

## 7. 人工复核的真实质量（如实记录，不美化）

- 验收线定的是一致率>90%，**实测只有47.5%**；9位标注员一致率从81.6%到16.8%不等，
  其中1位（Junwei Zheng）判定结果跟随机蒙的重合率过高（43.6% vs 8选3随机期望约
  37.5%），整包判定作废不采信
- 标准答案指定为正确的选项中，**27.1%（769/2836）被人工逐项核对时明确标为"画面否定"**
- 一个反直觉的发现：某标注员总一致率36.4%（全场最低之一），但只看他自评`clear`
  （有把握）的题，一致率反而是全场最高的83.0%——**总一致率低不等于标注质量差**，
  要结合标注员自己的把握度一起看

## 8. 文件索引

| 内容 | 路径 |
|---|---|
| 标注者操作手册（中/英，仓库根目录） | `校验操作手册.md` / `JUDGING_MANUAL.md` |
| 出题流水线总纲（含五道闸门） | `qa_fix_gold_recovery_and_adjudication/docs/FLOODGATE_ANNOTATION.md` |
| 多片段流水线规范（八道闸门+防泄漏） | `multi_video_cross_video_methodology/source_docs/ANNOTATION_AUDIT_GUIDE.md` |
| 多片段出题代码 | `multi_video_cross_video_methodology/source_docs/generate_cross_video_qa.py` |
| 语料二核验总结+分类方法 | `segments_60s_gemini_verification/docs/VERIFICATION_SUMMARY.md` / `MY_CLASSIFICATION_SUMMARY.md` / `question_type_spec.md` |
| 复核回收与标注员质量分档 | `qa_fix_gold_recovery_and_adjudication/docs/GOLD_RECOVERY_README.md` |
| 基准设计评审（含闸门通过率统计表） | `qa_fix_gold_recovery_and_adjudication/docs/BENCHMARK_DESIGN_REVIEW.md` |
