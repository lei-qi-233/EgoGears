# RunningBench 人工复核集：答案恢复 + 三方裁决 + 修复（1526 → 1487）

**这也不是我这次做的工作**——是项目内部这几天（2026-09-14~17）刚做完的一条完整流水线，我只做了核实、汇总和搬运，
`docs/`、`code/`、`data/` 里全是原始文件，未做任何删改。核实过每份文档都有对应代码可以逐行印证，时间戳互相吻合。

**和另外两个文件夹的关系**：
- `segments_60s_gemini_verification/` 是我自己的工作（`segments60s_kit` 6452 题的分类+核验），题源不同，不重叠。
- `multi_video_cross_video_methodology/` 讲的是**这 1526 题从哪来**（人工出题+跨视频生成的方法论）。
- **这个文件夹讲的是这 1526 题在交付之后经历了什么**——答案怎么弄丢又找回来的、Gemini/人工/旧gold三方怎么互相校验、
  查出问题的题怎么带视频修复、最后怎么收敛成 1487 题可用语料。是同一批题的下一个阶段。

---

## 1. 起点：为什么标准答案要"恢复"

10 个题包（`annotation_bundles/`，1526 题）分发出去时**不含标准答案**——校验方式设计成"回收方自己拿标准答案去对"。
但那批标准答案随造包的机器（`5hk3xk6dxy`）一起丢了：2026-09-09 该机器 FAILED 结束，`copy_artifacts`、`logs`
均为空，仅存的代码备份全文搜不到任何 `review_id`/`RBMA`/`bundle_0` 痕迹——**备份只留了代码，没留数据**。

题包用了一套全新的 `review_id`，和恢复出来的旧语料（11,802题全量/5,971题校验包/8,026题curated segments）**交集为零**，
但题目文本本身是旧的，只是重新编了号。恢复方法：拿题干原文去旧语料里查，比对选项文本集合是否完全一致，一致就把
旧答案的字母按选项文本重映射到新题包的字母（因为渲染时选项按 `Object.keys().sort()` 排序，字母空间是确定的）。

**恢复结果**：1,241 题原文匹配恢复、257 题（`rbma273`那批，旧语料里本来就零命中）用 Gemini 独立重判补出候选答案、
28 题彻底找不到（选项被改写过/模型三票钉不住/重试仍无结论）。

> `derived_unverified` 那 257 题不能当 gold 用——拿人工判定外部校准，exact 只有 39.7%，而且"模型三票是否一致"
> 完全不预测对错（三票全一致 exact 39.4%，三票有分歧反而 40.5%）。整条语料从 caption 到出题到闸门全程用的都是
> Gemini 家族模型，现在又用同族模型判答案，三票一致反映的是**自我一致**，不是被画面钉住了。

---

## 2. 三方交叉校验：gold / 人工 / Gemini，谁也不能自证

`code/adjudicate.py` 开头这句话是整套裁决逻辑的核心：

> **"gold 只有在两个互相看不到对方结果的独立视角一致反对它时才会被推翻。两个视角本身有分歧时，
> 交给第二轮、字母重新打乱的 Gemini 投票去裁决，而不是谁声音大听谁的。"**

三个视角：
- **gold**——从旧语料恢复出来的标准答案（来源已记录）
- **人工**——标注员看了视频、不知道 gold 是什么
- **Gemini 视频复核**——`gemini-3.1-pro-preview` 看了同样的视频、同样不知道 gold

外加一道完全不给视频的**盲猜闸门**。

### 三方谁跟谁更一致

| 对比 | 一致率 | 样本 |
|---|---:|---:|
| Gemini视频复核 vs 恢复的gold | 77.7% | n=1241, 95% CI [75.3, 79.9] |
| gold标为正确的选项中被Gemini画面判"否定" | 12.6% | 339/2686 |
| Gemini视频复核 vs 人工（互不知情） | 46.5% | n=1244 |

### 人工标注的严格度差 40 倍，且和"哪个包分给谁"完全绑定

每个标注员正好对应一个 bundle，所以按包统计的缺陷率其实就是标注员效应：Yufan Chen 只标 3% ambiguous，
Di 标 54% ambiguous（`QA_FIX_REPORT.md` 有完整10人分布表）。

**`Junwei Zheng` 的提交被判定为不可信**（`code/adjudicate.py` 里硬编码 `UNTRUSTED = {"Junwei Zheng"}`），
裁决时不计入确认票。两条独立证据：选项重合率 43.6%，8选3的随机期望约37.5%，几乎没有信号；
且在"对旧gold"和"对模型独立重判"两套互不相关的参照系下都垫底（16.8% / 0/26）。

反过来，`Di` 的低总一致率（36.4%）不代表标注质量差——只看 Di 自己判定为 `clear` 的题，一致率83.0%，
全场最高。**Di 是在诚实标注 ambiguous，不是在敷衍。** `code/adjudicate.py` 里为此专门维护了一份
`TRUSTED_HUMAN` 名单，用"该标注员说 clear 时到底准不准"这个更细的指标决定权重，而不是简单看总分。

### 盲猜闸门

三票取二判 guessable：**14.0%**（174/1241），验收线是 15%，**压线通过**。项目历史语料是 48.6%。

---

## 3. 裁决：15 种判决类型

`code/adjudicate.py` 的 `decide()` 函数为每题产出 15 种判决之一，`code/build_v2.py` 把它们分四组：

| 分组 | 判决类型 | 处理 |
|---|---|---|
| **KEEP**（沿用原答案） | `KEEP_CONFIRMED` `KEEP_GOLD_HUMAN_CONFIRMS` `KEEP_GOLD_HUMAN_UNSURE` `KEEP_GEMINI_ONLY` `KEEP_CONFIRMED_CROSSMODEL` `KEEP_GOLD_CROSSMODEL` `KEEP_GOLD_HUMAN_DECIDES` | 直接进最终语料 |
| **REKEY**（改答案） | `REKEY` `KEY_FROM_AGREEMENT` `KEY_FROM_GEMINI2` `REKEY_GEMINI_ONLY` `REKEY_CROSSMODEL` `REKEY_HUMAN` | 换成新答案，进最终语料 |
| **DROPPABLE**（送修或淘汰） | `BROKEN_NO_CORRECT_OPTION` `GOLD_CONTRADICTED_NO_REPLACEMENT` `UNRESOLVED` `NO_GOLD_UNRESOLVED` `NEEDS_VOTE2` `PENDING_VIDEO` `UNRESOLVED_MODELS_FAVOR_GOLD` | 先送 `code/repair.py` 带视频修，修不好才淘汰 |

`KEEP_CONFIRMED`（449题，29.4%）最多——两个互不知情的视角刚好落在同一个答案上。`UNRESOLVED`类只占约10%。

---

## 4. 修复：带视频改选项，最多三轮

对 `DROPPABLE` 那批题（582道）用 `code/repair.py` 带视频重造干扰项/改答案，**每轮必须重新过两道验收**
（不给答案的视频复核 + 不给视频的盲猜闸门），过不了就退回原题再修一轮：

```
轮次分布  {1轮: 319, 2轮: 111, 3轮: 66, 4轮: 7, 5轮: 1}
修复结果  送修582 → 修好504（86.6%）
```

修不好的主因：改完了模型盲猜还是能猜中（56次）、视频复核后模型给出的答案和标注答案对不上（多个具体选项各6-12次）。

---

## 5. 最终交付：1526 → 1487

`code/build_v2.py` 把上面所有判决收敛成四种最终状态：

| 状态 | 题数 | 占比 |
|---|---:|---:|
| confirmed | 706 | 46.3% |
| repaired | 504 | 33.0% |
| rekeyed | 277 | 18.2% |
| **dropped** | **39** | **2.6%** |

**可用题库 1,487 题** → `data/runningbench_v2_kept.jsonl`；全量 1,526 题（含每道淘汰题的淘汰理由）→ `data/runningbench_v2.jsonl`。

淘汰的 39 题里，13 题是修复后仍能被盲猜、11 题是两方复核吵不出结果、7 题是两个模型都支持gold但人工不认可、
4 题是gold被画面否定且找不到替代答案、3题选项里没有一个是对的、1题镜头压根没拍到问题问的地方。

---

## 6. 一个缺口——**更新：已经被补上了**

`docs/FLOODGATE_ANNOTATION.md` 第22行写明：整片视频出题的主流程脚本 `build_fullvideo_annotations.py`
靠 `load_generator()` 动态加载 `repair_runningbench_annotations.py`，从那里取 `Floodgate`/`CAPTION_MODEL`/
`STRUCTURE_MODEL`/`clean_json`/`generate_valid_json` 这几个符号。**`repair_runningbench_annotations.py`
这个文件本身没有备份到 conductor**，当时的结论是"从零开始按最初流程重新出题"这一步跑不起来。

**这个结论现在过时了。** `../model_evaluation_harness/code/floodgate.py`（属于本仓库另一个新增文件夹，
是给模型跑评测用的客户端）开头写着："这是 `build_fullvideo_annotations.py` 曾经从
`repair_runningbench_annotations.py` 动态加载的那个模块，那个文件从没备份到 Conductor。"
——**换句话说, 缺失的符号已经被按 `FLOODGATE_ANNOTATION.md` 第2节的思路重建出来了**，虽然重建的目的
是给评测用，不是给出题用，但缺口本身不再是"跑不起来"，是"有等价实现，只是路径变了"。

---

## 6.5 一个真实的风险：这批题带着标准答案，和还在盲判的标注包放在同一个仓库

`code/upload_v2_to_hf.py` 自己生成的仓库说明文本里写着这句话：

> ⚠️ **本目录含标准答案。** `annotation_bundles/` 的题包是刻意不含答案的，标注员在不知道答案的前提下盲判。
> **只要还有标注员在做题，就不要把本目录的访问权给他们。**

这不是我编的，是这份代码自己的设计意图。核实过：仓库本身是 **private**（不是公开的），风险不是"网上任何人
都能看到"，但如果标注员是通过直接访问这个 HF 仓库来下载自己那份 `bundle_XX.tar` 的（README 里没写清楚
分发方式是"给仓库权限"还是"单独把tar发过去"），他们对仓库的浏览权限就和这份带答案的数据是同一级——
翻一下目录就能看到。而 `annotation_Haiwen_Sun_2026-09-16.json` 是这几天刚回传的，说明**标注工作可能还没结束**。
**这件事需要你确认两点**：①标注员拿到题包的方式是不是直接访问这个仓库 ②标注是否已经全部收完。
两条只要有一条是"是/否"（还在标、且靠仓库权限分发），就该把带答案的这份数据挪走，而不是留在原地。

---

## 7. 复算方式

```bash
cd code/
python3 video_review.py --vote 0 --workers 14 --rps 0.45   # 不给答案, 看视频独立复核
python3 blind_gate.py --votes 3                              # 完全不给视频, 盲猜闸门
python3 adjudicate.py                                         # 三方交叉裁决, 产出15种判决
python3 repair.py --decisions BROKEN_NO_CORRECT_OPTION,GOLD_CONTRADICTED_NO_REPLACEMENT
python3 repair.py --guessable-only                            # 带视频修复, 每轮重新过两道验收
python3 build_v2.py                                            # 收敛成 confirmed/repaired/rekeyed/dropped
```

依赖 `data/master_index.jsonl`（1526题权威索引，和 `multi_video_cross_video_methodology/` 里那份是同一个文件），
以及每题对应的视频片段——原片在 `annotation_bundles/bundle_01~10.tar`（仓库根目录，21GB，3812个mp4，团队原来就传的）。

**`video_review.py`跑之前实际上还有一步**：`code/prep_clips.py`把每题的片段压到能inline提交给Gemini的体积
（≤13MB原始→≤17.3MB base64，压到20MB硬上限以下）。关键设计：480p不降（360p会让模型把清晰路牌读成"模糊"），
降的是帧率——Gemini大约按1fps采样视频，1526题里863题不用处理直接能塞进预算，剩下663题超预算的把fps降到5就够。
另有 `code/merge_overlimit_clips.py` 处理一个边界情况：Floodgate上所有模型都拒绝超过10个视频文件的请求，
本语料有3道`cross_video`题挂了11-12个片段，靠合并同一录像标签下的多个片段（不丢信息）把数量压到10以内，
不然这3题会一直卡在`not_reviewed`，看着像"没人做"而不是"这题结构性跑不了"。

---

## 8. 文件清单

```
docs/
  DATASET_AUDIT_README.md      整体数据集审计说明: 两个视频来源/因子网格/已知混淆因子(音频=参与者捷径等)
  BENCHMARK_DESIGN_REVIEW.md   基准设计评审记录(148段录像的完整现状, 写给要改设计的人看)
  FLOODGATE_ANNOTATION.md      Gemini出题+五道闸门的完整调用流程, 含第6节的已知缺口说明
  GOLD_RECOVERY_README.md      本README §1的信息来源: 答案怎么丢的、怎么恢复的、恢复结果的可信度分档
  QA_FIX_REPORT.md             本README §2-5大部分数字的来源: 三方一致率/15种判决分布/修复结果
  JUDGING_MANUAL.md            人工标注员操作手册(英文版)
  校验操作手册.md                同上, 中文版, 和HF仓库根目录README.md里嵌入的手册是同一份

code/
  video_review.py    不给答案, 独立看视频复核(人工/Gemini两方各跑一次, 互不知情)
  blind_gate.py       完全不给视频的盲猜闸门, 三票取二
  adjudicate.py       三方交叉裁决核心逻辑, 15种判决类型的定义都在这里, 含Junwei Zheng不可信判定
  repair.py           带视频修复逻辑, 每轮重新过两道验收, 最多三轮
  build_v2.py         最终收敛: 15种判决→4种状态→1487题可用语料
  floodgate.py        调Gemini的公共层
  prep_clips.py            切片预算压缩: 480p+降帧率, 塞进inline上传的20MB上限
  merge_overlimit_clips.py 合并超过10个片段的题(Floodgate的硬限制), 让3道卡住的cross_video题能跑
  analyze.py           交叉制表: gold x 人工 x Gemini视频复核 x 盲猜闸门四个视角两两对照
  audit_rounds.py      独立复核"修复轮数越多是不是越靠运气而不是真的修好了"(多次重采样的多重比较担忧)
  peek.py              把某道题Gemini复核引用的具体帧抽出来, 给一双非Gemini的眼睛看
  report.py            QA_FIX_REPORT.md的生成器——报告里每个数字都能靠重跑这个脚本复算
  upload_v2_to_hf.py   把v2语料发布到HF的脚本(未实际跑过, v2/前缀在仓库里还不存在); 硬编码的token路径
                       指向另一个session的scratchpad, 是失效引用, 不是活凭证; 见第6.5节的风险提示
  download_videos.py   1526题人工复核CLI工具包的下载器(和JUDGING_MANUAL.md配套, 仓库里还有英文版
                       download_videos.py/manual_verify.py放在另外两个目录, 内容基本对应, 未重复上传)
  manual_verify.py     同上CLI工具包的复核程序本体

data/
  master_index.jsonl          1526题权威索引(输入)
  runningbench_v2.jsonl        全量1526题, 含每道淘汰题的淘汰理由(输出, 全量)
  runningbench_v2_kept.jsonl   可用语料1487题(输出, 主交付物, 见第6.5节的访问风险提示)
```

原始位置：`/mnt/data/cvhci_video_understanding/{README.md, FLOODGATE_ANNOTATION.md, BENCHMARK_DESIGN_REVIEW.md,
hf_review_recovery/, qa_fix/}`。代码和文档时间戳均为 2026-09-14~17，是这几天刚完成的工作。
