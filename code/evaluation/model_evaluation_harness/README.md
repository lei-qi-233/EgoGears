# RunningBench 模型评测harness + 24模型真实成绩

**这也不是我这次做的工作**——是项目内部已经跑完的一套完整评测流水线，我只做了核实和搬运。
`code/`、`results/`、`analysis.json` 全是原始文件，未做任何删改。

**和另外三个文件夹的关系**：前三个都是"把 RunningBench 这套题本身弄干净"（分类/核验/答案恢复/裁决），
**这个文件夹是"拿弄干净之后的题去真的评测模型"**——评测对象是 `qa_fix_gold_recovery_and_adjudication/`
里那 1487 题（`runningbench_v2_kept.jsonl`），24 个视觉语言模型已经跑出了真实成绩。

---

## 1. 一眼看清：这套基准还远没被打穿

| 排名 | 模型 | exact% | overlap% | wellformed% |
|---|---|---:|---:|---:|
| 1 | Qwen3-VL-235B-A22B-Thinking | 38.4 | 76.3 | 72.2 |
| 2 | GLM-4.6V | 36.9 | 68.6 | 82.5 |
| 3 | Gemma-4-31B-it | 33.1 | 59.9 | 99.7 |
| 4 | Qwen3-VL-235B-A22B-Instruct | 28.6 | 55.6 | 99.6 |
| 5 | Qwen2.5-VL-72B-Instruct | 26.4 | 53.5 | 99.7 |
| ... | （完整24个模型见`analysis.json`） | | | |
| 末位 | ERNIE-4.5-VL-28B-A3B | 4.0 | 67.8 | 40.1 |

**最强模型精确匹配也只有 38.4%**——这套基准（1487题，5种unit：whole_video/cross_video/cross_recording/
multi_caption_multi_clip/anonymous_multi_clip，和`multi_video_cross_video_methodology/`里讲的是同一套
分类）目前是真的难，不是随便一个模型就能刷高分。

`wellformed%` 这一列值得留意——不是所有低分都代表"看不懂视频"：`ERNIE-4.5-VL-28B-A3B` 只有40.1%的输出
格式合规（很可能是返回格式跑偏，被当成答错），`Qwen3-VL-235B-A22B-Thinking`/`Qwen3-VL-30B-A3B-Thinking`
的wellformed也明显低于其它模型（70%左右）——"思考型"模型似乎更容易在格式合规上栽跟头，拿到的exact分数
可能被这个因素拖累了，不完全是视觉理解能力的差距。

---

## 2. 评测方法论——三个防作弊设计

**`run_eval.py`**——统一评测入口，两个后端（Floodgate跑Gemini家族 / OpenAI兼容接口跑vLLM服务的开源模型），
同一套prompt、同一套评分。**关键设计**：每道题的选项字母按题目id做种重新打乱，再映射回真实字母——
不这么做的话，一个偏爱选"B"的模型会凭空拿到高于随机水平的分数，这个偏差和视频理解能力毫无关系。

**`extract_frames.py`**——帧预算是**按题分配，不是按视频分配**。RunningBench的题目携带1到10个片段不等，
如果按视频固定抽帧数，一道10片段的题会拿到一道单片段题十倍的证据量，模型间的抽帧策略一旦不同，
成绩就没法比。所以预算是"每题固定64帧"，按各片段时长比例切分，同时设了每片段最少4帧的下限防止某个
片段被抽没。所有模型看的是完全相同的一组JPEG。

**`score.py`——正确的随机基线不是1/n。** RunningBench混合了3~8个选项、1~5个正确答案，随机基线是
`C(选项数, 正确数)`的组合数分之一，逐题算完再对模型实际答过的题取平均，不是笼统给一个数。不这样算，
一个"偏爱单选题"的模型会显得比实际强——单选题的组合基线天然比多选题高。

**`build_analysis.py`**——把各家模型的公开元数据(总参数/激活参数/是否MoE/是否thinking模式/发布年份)
和成绩拼起来，产出`analysis.json`里的这套排行榜，能看出"参数量/推理模式和分数的相关性"。
**`merge_shards.py`**——大模型分片跑(`.shardKofN.jsonl`)之后合并回单个结果文件用于计分。
**`floodgate.py`——见下方重要说明。**

---

## 3. 一个重要的更正：`FLOODGATE_ANNOTATION.md` 点名缺失的模块，已经被重建了

`qa_fix_gold_recovery_and_adjudication/docs/FLOODGATE_ANNOTATION.md` 第22行写着：
`build_fullvideo_annotations.py` 曾经动态加载 `repair_runningbench_annotations.py`，取
`Floodgate`/`CAPTION_MODEL`/`STRUCTURE_MODEL`/`clean_json`/`generate_valid_json` 这几个符号，
但那个文件从未备份到 conductor，被记录为"跑不起来"的缺口。

`code/floodgate.py` 的文件头写的原话：

> "这是 `build_fullvideo_annotations.py` 曾经从 `repair_runningbench_annotations.py` 动态加载的
> 那个模块，那个文件从没备份到 Conductor。"

**换句话说，这个模块已经按 `FLOODGATE_ANNOTATION.md` 第2节的思路重建出来了**——虽然重建的目的是给
评测用（不是给最初的出题流程用），但"缺失符号=跑不起来"这个结论现在已经不成立，是"有等价实现，
路径变了"。如果需要按最初的出题流程复现，这个文件是现成的起点。

---

## 4. 复现

```bash
cd code/
python3 extract_frames.py                                    # 每题按比例抽帧, 64帧预算/题
python3 run_eval.py --model gemini-3.1-pro-preview --backend floodgate
python3 run_eval.py --model Qwen2.5-VL-7B --backend openai --base-url <vLLM服务地址>
python3 merge_shards.py <model>                               # 分片跑的话先合并
python3 score.py --model <model>                               # 输出exact/overlap两个指标
python3 build_analysis.py                                     # 汇总成 analysis.json 排行榜
```

依赖 `qa_fix_gold_recovery_and_adjudication/data/runningbench_v2_kept.jsonl`（1487题评测对象）
以及对应视频片段（`annotation_bundles/`）。

---

## 5. 文件清单

```
code/
  run_eval.py          评测主入口, 两个后端(Floodgate/OpenAI兼容), 选项字母防位置偏差重打乱
  extract_frames.py    按题(非按视频)分配64帧预算, 所有模型看同一组JPEG
  score.py             两个指标(exact/overlap) + 正确的组合数随机基线
  build_analysis.py    拼模型元数据(参数量/MoE/thinking/年份)产出analysis.json排行榜
  merge_shards.py      合并分片评测结果
  floodgate.py         Gemini客户端, 重建了FLOODGATE_ANNOTATION.md点名缺失的模块(见第3节)

results/   24~30个模型的原始逐题预测(<model>.jsonl, 每题一行, 含pred/gold/exact/overlap字段)
analysis.json   汇总排行榜: 24个模型的exact/overlap/wellformed/按unit拆分成绩 + 模型元数据
```

原始位置：`/mnt/data/cvhci_video_understanding/eval/`（含本次未上传的 `frames/`、`bundle/`、`orchestrate/`、
`qa_fix_stub/` 等辅助目录，多为运行时产物或路径桩，非核心代码）。
