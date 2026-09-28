# RunningBench 题目可行性复核与修复

对象：HF `taryya/RunningBench`，10 个题包共 **1526 题**。方法：**每一题都用 Gemini floodgate
看真实视频逐选项复核**，与恢复出的标准答案、人工标注三方交叉，再对判定为坏的题做带视频的修复。

复核覆盖：视频复核 1526/1526，盲猜闸门 1526/1526。

## 1. 三个视角互相怎么看

- Gemini 视频复核 vs 恢复的 gold：**77.7%** 完全一致（n=1241，95% CI [75.3, 79.9]）
- gold 指定为正确的选项中，被画面判为**否定**的占 **12.6%**（339/2686）
- Gemini 视频复核 vs 人工（两边都看了视频、都不知道 gold）：**46.5%** 完全一致（n=1244）

## 2. 人工标注的严格度差 40 倍，而且和题包一一对应

每个标注员包一个 bundle，所以按 bundle 统计的缺陷率其实是标注员效应。

| 标注员 | 题数 | clear | ambiguous | cant_tell |
|---|---:|---:|---:|---:|
| Di | 154 | 38% | **54%** | 8% |
| Xiaoye Wang | 74 | 59% | **32%** | 8% |
| Lei | 153 | 60% | **31%** | 8% |
| Chengzhi Wu | 152 | 64% | **30%** | 6% |
| Ruiping Liu/ Qian Yin | 150 | 77% | **14%** | 9% |
| Haiwen Sun | 105 | 92% | **8%** | 0% |
| Chen Zhang | 151 | 75% | **7%** | 18% |
| Yufan Chen | 151 | 97% | **3%** | 1% |
| zhihang chen | 154 | 96% | **1%** | 3% |
| Junwei Zheng | 151 | 93% | **1%** | 7% |

其中 **40 条** ambiguous 配的是同一句英文模板，不是逐题观察。

## 3. 判决

| 判决 | 题数 | 占已判 |
|---|---:|---:|
| `KEEP_CONFIRMED` | 449 | 29.4% |
| `REKEY_HUMAN` | 235 | 15.4% |
| `KEEP_GEMINI_ONLY` | 159 | 10.4% |
| `KEY_FROM_GEMINI2` | 156 | 10.2% |
| `KEEP_GOLD_HUMAN_UNSURE` | 151 | 9.9% |
| `KEY_FROM_AGREEMENT` | 91 | 6.0% |
| `UNRESOLVED` | 75 | 4.9% |
| `KEEP_GOLD_HUMAN_CONFIRMS` | 61 | 4.0% |
| `GOLD_CONTRADICTED_NO_REPLACEMENT` | 38 | 2.5% |
| `BROKEN_NO_CORRECT_OPTION` | 31 | 2.0% |
| `REKEY` | 26 | 1.7% |
| `NO_GOLD_UNRESOLVED` | 23 | 1.5% |
| `REKEY_GEMINI_ONLY` | 17 | 1.1% |
| `UNRESOLVED_MODELS_FAVOR_GOLD` | 7 | 0.5% |
| `KEEP_GOLD_CROSSMODEL` | 7 | 0.5% |

## 4. 盲猜闸门（完全不给视频）

三票取二判 guessable：**14.0%**（174/1241，95% CI [12.2, 16.1]）。
项目历史语料是 48.6%，验收线是 15%。

| 题型规格 | 可盲猜 | 随机基线 |
|---|---:|---:|
| 8 选 3 | 15.5% (92/593) | 1.8% |
| 6 选 1 | 13.0% (63/485) | 16.7% |
| 8 选 2 | 12.0% (14/117) | 3.6% |
| 8 选 4 | 9.1% (4/44) | 1.4% |
| 8 选 5 | 0.0% (0/1) | 1.8% |
| 8 选 7 | 100.0% (1/1) | 12.5% |

## 5. 修复

送修 582 题，修好 **504** 题（86.6%）。
每题最多三轮，每轮都必须重新过两道验收（不给答案的视频复核 + 不给视频的盲猜闸门），过不了就退回原题。
轮次分布：{1: 319, 2: 111, 3: 66, 4: 7, 5: 1}

修不好的原因：
- still blind-guessable  — 56 次
- video review answered ['A' — 12 次
- video review answered ['B'] — 10 次
- video review verdict ambiguous — 8 次
- video review answered ['B' — 8 次
- video review answered ['D'] — 6 次

## 6. 交付

| 状态 | 题数 | 占比 |
|---|---:|---:|
| confirmed | 706 | 46.3% |
| repaired | 504 | 33.0% |
| rekeyed | 277 | 18.2% |
| dropped | 39 | 2.6% |

淘汰原因：

| 原因 | 题数 |
|---|---:|
| blind_guessable_unrepaired | 13 |
| unresolved_disagreement | 11 |
| models_favor_gold_human_dissents | 7 |
| gold_contradicted_no_replacement | 4 |
| no_correct_option | 3 |
| footage_does_not_cover | 1 |

可用题库 **1487** 题，文件 `runningbench_v2_kept.jsonl`；全量含淘汰理由在 `runningbench_v2.jsonl`。

## 7. 复算方式

```
python3 prep_clips.py 48       # 把每题的片段压进 13 MB inline 预算
python3 video_review.py --vote 0 --workers 14 --rps 0.45
python3 blind_gate.py --votes 3
python3 adjudicate.py
python3 repair.py --decisions BROKEN_NO_CORRECT_OPTION,GOLD_CONTRADICTED_NO_REPLACEMENT
python3 repair.py --guessable-only
python3 build_v2.py && python3 report.py
```
