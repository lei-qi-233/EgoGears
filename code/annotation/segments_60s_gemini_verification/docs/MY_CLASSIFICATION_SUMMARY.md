# RunningBench 题型分类总结（自建 9 类口径）

**注**：本文档说的是我自己搭的这套分类（`report_class`），不是后来发现的官方 5 大类/18 小类 `qa_taxonomy.json`。两者是并行的两套体系，本文档只讲这一套，以及基于它正在跑的核验工作。

---

## 1. 为什么这么分

`segments60s_kit/README.md` 写明："`captions/segments_60s.jsonl` 里那 1,400 条描述就是出题依据"——题目是照 caption 的字段模板生成的。所以**一道题的类型 = 它派生自 caption 的哪个字段**，而不是靠看题干措辞去猜（题干是模板生成的，猜措辞等于把生成模板的痕迹当标签）。

caption 有 5 个字段，映射关系：

| caption 字段 | 对应类别 |
|---|---|
| `overall_environment` | Scene and Place QA |
| `objects_and_attributes` | Object and Attribute QA |
| `spatial_relations` | Ego-relative Spatial Relation |
| `trajectory_and_turnings` | Trajectory-grounded QA / Ego-motion Recognition（按覆盖整段还是局部机动区分） |
| `actions_and_events` | Ego-motion Recognition / Actor Action and Motion（按动作主语是拍摄者还是外部对象区分） |

另外 3 类不对应 caption 字段，是**提问形式**，判定优先于字段归因：

- **Text and OCR** —— 题干要求读路牌/车身/建筑上的文字
- **Temporal Grounding** —— 必须给出具体时刻才能答对（用"抹掉时间戳后选项还分不分得开"做必要性测试，防止把普通轨迹题误判成这一类）
- **Event Sequencing** —— 区分点是顺序（选项是同一组元素的不同排列）

判定优先级（自上而下，先命中者胜）：Text/OCR → Temporal Grounding → Event Sequencing → 自身运动(Trajectory/Ego-motion) → 外部对象动作(Actor) → 方位(Spatial Relation，用干扰项是否在 caption 里出现来判) → 场景(Scene) → 兜底(Object/Attribute)。

**验证过的准确性**：用 46 题分层人工标注做 gold，这套规则分类和 gemini-3.1-pro-preview 文本分类的一致率都是 **96%**，但规则分类**零 API 调用、免费、秒级跑完全量**。

---

## 2. 9 个类别的分布（6452 题可用池）

排除了 318 题视频过短/被截断的（片段区间超出视频真实结尾，或可见时长<15秒）之后的可用池。

| 类别 | 题数 | 占比 | 覆盖视频 |
|---|---:|---:|---:|
| Object and Attribute QA | 2496 | 38.7% | 120 |
| Ego-relative Spatial Relation | 1942 | 30.1% | 119 |
| Trajectory-grounded QA | 1061 | 16.4% | 119 |
| Ego-motion Recognition | 441 | 6.8% | 107 |
| Temporal Grounding | 151 | 2.3% | 62 |
| Scene and Place QA | 141 | 2.2% | 69 |
| Event Sequencing | 118 | 1.8% | 60 |
| Actor Action and Motion | 80 | 1.2% | 55 |
| Text and OCR | 22 | 0.3% | 15 |
| **合计** | **6452** | 100% | 126（并集） |

分类标签和视频身份基本独立（归一化互信息 NMI ≈ 0.04），每个 60s 片段平均命中 3 个类别——说明标签携带的是题目本身的信息，不是"这个视频天生只出这类题"，按类别拆分数不会被"哪些视频难"这个混杂因素污染。

---

## 3. 分类之上还叠了一层：视觉核验

分类只回答"这道题考什么"，不回答"这道题的标准答案对不对"。`segments60s_kit/README.md` §6 说得很清楚：这 6770 题**只过了盲猜闸门，视觉核验从未跑过**，同语料其它层实测淘汰率 40.7%。所以光有分类不够，每道题还得对着画面核对一遍选项。

核验方式：60 秒片段切片（480p/crf30/去音轨）→ inline 发给 `gemini-3.1-pro-preview` → 逐选项输出 `supported`/`ruled_out`/`undecidable`，整题给出 `answer_correct`/`answer_wrong`/`insufficient`。**不用 caption 核验**——题目就是照 caption 生成的，拿 caption 核是循环论证，检不出真错误。

### 核验进度：已 100% 完成（全量普查，不是抽样估计）

| 阶段 | 状态 |
|---|---|
| 第一轮抽样核验 | 1089 题，已完成 |
| 第二轮补抽核验 | 462 题，已完成 |
| 全量核验剩余池 | 4901 题，已完成（12:59 全部跑完） |
| **合计** | **6452/6452，6452 题可用池 100% 普查** |

**整体通过率（精确值，非抽样估计）**：5081/6452 = **78.8%**（`answer_correct` 5081、`answer_wrong` 1364、`insufficient` 7）。

已完成：
1. ✅ 用完整数据重算了 9 类各自的**精确**通过率（不再是抽样外推）
2. ✅ 重新扫描每视频抽样上限（通过池从 1211 涨到 5081 后，最优值仍是 15）
3. ✅ 重建了最终 698 题子集，9 类全覆盖
4. ✅ 更新了 `VERIFICATION_SUMMARY.md`（全量普查版）

---

## 4. 已知局限（如实说明）

- **本文档的类别占比是 6452 题池的静态占比**，不是最终 700 题子集里的占比——子集会按这个占比分配大类配额，小类给固定下限（~21题），具体见 `VERIFICATION_SUMMARY.md`。
- **9 类里 `Text and OCR` 和 `Temporal Grounding` 是我自建的，不在官方 `qa_taxonomy.json` 的 18 个子类里**；反过来官方有的 `Counting QA`、`Negative or Absence QA`、`Event Memory QA`、`Action Repetition`、按行人/车辆拆分的 Actor 类、整个 `Robustness QA` 大类，我都没有用——这两套分类目前是平行的，还没有做映射。
- **核验只用了单一模型**（gemini-3.1-pro-preview），没有多模型交叉或人工复核，核验本身的错误率未知。
- **`Ego-motion Recognition` vs `Trajectory-grounded QA` 的边界**依赖"覆盖整段还是局部机动"这条启发式，46题验证里这条边界样本量偏少，是这套分类里置信度最低的一处。
