# RunningBench 60s 片段题型分类规范 v4

**适用数据**：`segments60s_kit/questions/segments_60s.jsonl` — 6770 题 / 1400 个 60s 片段 / 126 个视频（每片段中位 5 题）。
**数据性质**：第一人称移动相机的室外步行与跑步视频，无手部操作类内容。
**产物**：`segments_60s_typed.jsonl`（原题 + 三个新字段）、`classify_question_types.py`（规则分类器）。

---

## 1. 总则

**单标签。** 每题恰好一个 `type_sub_category`，`type_category` 由它唯一决定。

**判定依据是「正确项与干扰项的区分点」，不是题干措辞。**
标注前先做干扰项分析：把正确项和最接近的干扰项并排看，问一句「答对这题，最少需要从视频里读出什么信息」。那个信息的类型就是标签。

这条是本规范最重要的一条，因为**题干是模板生成的，措辞不携带标注信息**。例如 `c02d5b42` 的题干写着 "trajectory and spatial relations"，但两个正确项（C、G）全是自身运动，空间关系只出现在干扰项里 —— 该题判 Trajectory，不判 Spatial Relation。

---

## 2. 分类表

| sub_category | category | 题数 | 占比 | 覆盖视频 |
|---|---|---|---|---|
| Static Detail QA | Object and attribute QA | 2564 | 37.9% | 126 |
| Ego-relative Spatial Relation | Spatial relation QA | 1311 | 19.4% | 115 |
| Trajectory-grounded QA | Ego-motion and trajectory QA | 890 | 13.1% | 122 |
| Ego-motion Recognition | Ego-motion and trajectory QA | 646 | 9.5% | 116 |
| Temporally-scoped Object QA | Object and attribute QA | 597 | 8.8% | 108 |
| Actor Action and Motion | Actor and object motion QA | 285 | 4.2% | 100 |
| Scene and Place QA | Object and attribute QA | 183 | 2.7% | 78 |
| Temporal Grounding | Temporal QA | 153 | 2.3% | 63 |
| Event Sequencing | Temporal QA | 141 | 2.1% | 68 |

### 各类定义

- **Static Detail QA** — 物体的存在、身份、颜色/材质/形状/数量/状态，60s 内任一时刻单帧即可判定。
- **Temporally-scoped Object QA** — 同上，但必须定位到特定时间窗，或需要比较不同时刻才能判定。
- **Ego-relative Spatial Relation** — 相对相机或路径的左/右/前/后。干扰项是「同一物体的错误方位」。
- **Trajectory-grounded QA** — 覆盖整段的路径形状或路线走向。
- **Ego-motion Recognition** — 某一处的局部机动：停、转、变焦、低头看路面。
- **Actor Action and Motion** — 外部的人/车/动物做了什么、朝哪个方向运动。
- **Scene and Place QA** — 场所类型、天气、时段、光照等整体场景判断。
- **Temporal Grounding** — 某事发生在第几秒或哪个区间。
- **Event Sequencing** — 事件或地标的先后顺序。

---

## 3. 判定优先级

多类竞争时自上而下，先命中者胜。

1. **Temporal Grounding** — 须通过**必要性测试**（见 5.1）。
2. **Event Sequencing** — 区分点是顺序（选项是同一组元素的不同排列）。
3. **Ego-motion and trajectory** — 答案描述拍摄者自身运动。覆盖整段判 Trajectory-grounded，仅一次机动判 Ego-motion Recognition。
4. **Actor Action and Motion** — 答案主语是外部对象且含运动动词。
5. **Ego-relative Spatial Relation** — 区分点是方位，判据见 5.2。
6. **Scene and Place QA**。
7. **Static / Temporally-scoped Object QA** — 按时间轴判据（5.3）二选一。

---

## 4. 辅助字段

### `temporal_scope`（全部 6770 题都有）

`static` 5176 题（76.5%） / `scoped` 1594 题（23.5%）。

判据同 5.3，但作为正交维度发到每一类上，用于回答「benchmark 里有多少题不需要看完这一分钟」。各类的 scoped 占比：

| 类别 | scoped 占比 |
|---|---|
| Temporally-scoped Object QA / Temporal Grounding | 100%（定义使然） |
| Ego-motion Recognition | 33% |
| Actor Action and Motion | 33% |
| Ego-relative Spatial Relation | 26% |
| Trajectory-grounded QA | 18% |
| Event Sequencing | 9% |
| Scene and Place QA | 5% |
| Static Detail QA | 0%（定义使然） |

### `mixed_skill_options`

部分多选题的各选项分别考不同技能，单标签对这类题不 well-defined。规则：主标签取正确项中占多数的技能，无多数时按第 3 节优先级；同时置 `true`，分类型报指标时剔除。
实测占比：multi 题 29%（6/21），single 题 8%（2/25）。

---

## 5. 三条关键判据

### 5.1 Temporal Grounding 的必要性测试

抹掉所有选项里的时间戳，若正确项仍唯一可判，则**不是** Temporal Grounding。

例：`ec9baa0f`「直行 → 约 0:25 右转 → 继续直行」，抹掉 0:25 后仍能靠转向和后续动作唯一确定 → 判 Trajectory-grounded 而非 Temporal Grounding。

没有这条测试，Temporal Grounding 会吞掉大量轨迹题。

### 5.2 Spatial Relation vs Object QA 的判据

题干带区域词（"on the left side"、"directly ahead"）**不足以**判 Spatial Relation。真正的判据是干扰项里的物体在视频中是否存在：

- 干扰项「存在但方位错」→ **Spatial Relation**
- 干扰项「根本不存在」→ **Object QA**

用**同视频**其他题的正确选项做证据池交叉验证（名词短语 token 重合 ≥2 个且 ≥40%），存在率 ≥0.6 判 Spatial Relation。

两个实测坑：

- **证据池必须是视频级，不能是片段级。** 片段级中位只有 4 道邻题，信号被压平，会把大量 Object QA 误判成「干扰项不存在」。切换池粒度这一项就让整体一致率从 54% 掉下来过。
- 选项是多物体拼接时该启发式失效，但这类题在优先级 2 已被 Event Sequencing 截走，不受影响。
- 证据不足（`distractor_presence` 为 null）时**不默认判 Spatial Relation**，落回 Object QA —— Spatial Relation 需要正证据。

### 5.3 Static vs Temporally-scoped 的判据

满足任一条判 `scoped`：

1. 题干限定时间窗（"during the first 50 seconds"、"in the final segment"、"from 00:41 to 00:59"、"before you reach the T-junction"）。
2. 时间表述在**选项之间取值不同**，构成区分维度（early / midway / late）。单一取值不算 —— `1d0dc9de` 只有正确项 C 含 "near the end"，干扰项是不存在的物体，时间不是区分点，判 `static`。
3. 正确项描述状态变化（transitions from X to Y、comes into view），必须比较两个时刻。

否则判 `static`。

---

## 6. 被否决的方案与原因

### 6.1 `Ego-action Recognition` 不适用

该子类在 Ego4D 语境下指拍摄者的手部-物体交互动作。本数据集是室外行走跑步，46 题人工样本中 **0 例**。

看起来像「动作」的内容分两处：拍摄者自身的停/转/变焦归 Ego-motion，外部对象的动作归 **Actor Action and Motion**（285 题，4.2%）。最初草案用 `Ego-action Recognition` 覆盖的就是后者，但主体标反了 —— 不是拍摄者，是画面里的人和车。

### 6.2 Object Existence 与 Visual Attributes 切不开

v2 曾把物体类拆成「存在性」和「属性」两类。实测该切分不成立：

| 状态 | 自动分类与人工标注一致率 |
|---|---|
| 拆成 Existence / Attributes | **54%** |
| 合并 | **91%** |

差的 37 个百分点几乎全是这两类互串。原因是数据性质 —— 绝大多数题的形式是「以下哪些『物体+属性』描述是对的」，存在性判断和属性判断绑定在同一个选项里，不存在稳定区分点。

**改用时间轴切分**（Static / Temporally-scoped）后一致率保持 91%，切分成立。这也是本规范选择这条轴的原因：它对 60s 片段 benchmark 更有诊断价值 —— 测的是「要不要看完这一分钟」。

### 6.3 Text and OCR 不设为独立类

只有 20 题、覆盖 14 个视频，撑不起分类型报表，并入 Object QA。

---

## 7. 质量与精度边界

### 分类不跟着视频走

- **类型与视频的归一化互信息 NMI = 0.043**，接近独立。
- 每个 60s 片段平均命中 3.09 个类别，85% 的片段覆盖 2–4 类。
- 除 Scene/Temporal 三个小类外，每类横跨 100–126 个视频。

这说明类型标签携带的是**题目本身**的信息而非视频的信息，按类型拆分数时各类分差不会被「哪些视频难」这个混杂因素污染。这是这套分类可用的关键前提。

### 已知精度边界

- 规则分类器与 46 题人工标注一致率 **91%**（42/46）。残余错误：Spatial Relation → Temporally-scoped ×2、Static → Actor ×1、Actor → Ego-motion ×1。
- **Ego-motion Recognition 的 9.5% 是全量最弱的数字**：46 题样本按旧标签分层，A 类样本几乎全落在 Trajectory 一侧，只有 1 例人工确认了 Ego-motion Recognition。若要依赖这条边界，需对 A 类做一次针对性抽样复核；否则只报大类 Ego-motion and trajectory（22.7%）。
- Scene and Place、Temporal Grounding、Event Sequencing 各 140–190 题、覆盖 63–78 个视频，够做分类型报表但不宜再细分。
- **Static Detail QA 仍占 37.9%**，是最大的一类。时间轴切分只从中移走了 8.8 个百分点，这一格内部仍缺区分度。

### 报分数时的注意事项

类型与 arity 强相关：Ego-relative Spatial Relation 71% 是多选，Trajectory-grounded 93% 是单选，Event Sequencing 99% 是单选。**多选题的部分对怎么算分，会系统性抬高或压低特定类别** —— 拆类型报表前须先把多选评分方式定死。

---

## 8. 文件

| 文件 | 内容 |
|---|---|
| `segments_60s_typed.jsonl` | 6770 题原文 + `type_category` / `type_sub_category` / `temporal_scope` |
| `classify_question_types.py` | 规则分类器，实现第 3、5 节 |
| `question_types_full_auto.jsonl` | 分类器原始输出，含 `distractor_presence` 中间量 |
| `question_types_relabeled_46.jsonl` | 46 题人工标注，分层抽样、46 个不同视频，用作验收基准 |
| `question_types_original6_relabeled.jsonl` | 最初 6 条草案标注在本规范下的重标结果 |
