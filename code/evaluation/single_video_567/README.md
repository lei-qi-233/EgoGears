---
language:
- en
task_categories:
- visual-question-answering
tags:
- video-question-answering
- evaluation
- runningbench
---

# RunningBench single-video QA: 567 human-usable questions

This folder is a portable evaluation release for the 567-question human-curated single-video subset. The 567 videos are **required**: upload the entire `videos/` directory alongside the QA file. A QA-only upload cannot reproduce this benchmark. This release is separate from the original 698-question single-video set and the 1,487-question mixed/video set.

**To run the benchmark, or to hand it to an AI coding agent, start with [`AGENTS.md`](AGENTS.md).** It is a step-by-step runbook covering download, verification, per-model smoke tests, full runs and deliverables. For a single configuration, one command is enough: `bash code/run_config.sh <config_id>`.

## Contents

| Path | Contents |
|---|---|
| `AGENTS.md` / `CLAUDE.md` | Runbook for running the full evaluation (CLAUDE.md imports AGENTS.md) |
| `code/configs_34.csv` | The 34 configurations: run ID, exact Hugging Face / API model ID, thinking mode, token budget, suggested tensor parallelism |
| `code/run_config.sh` | Starts vLLM for one configuration, runs all 567 questions, and scores them |
| `data/single_567_human_eval.jsonl` | 567 questions, one JSON object per line |
| `human_answer_repair_567/single_567_human_eval.jsonl` | Identical file at the path expected by the included runner |
| `videos/<question_id>.mp4` | Exactly 567 pre-cut single-video clips |
| `data/video_sha256.csv` | File size and SHA-256 for every clip |
| `data/source_manifest.json` | Source hashes and correction counts |
| `RunningBench_567题可用数据说明.md` | Human-curation scope and limitations |
| `code/unified_eval.py` | Unified runner and scorer |
| `code/model_matrix_567.csv` | Target model/configuration matrix from the multi-video evaluation |

Each question has an `id`, `question`, option map (`A`, `B`, ...), answer letter list, video metadata and `media_file` pointing to `videos/<id>.mp4`. There are 325 single-select and 242 multi-select questions. The source comprises 562 complete clear human annotations and five repaired incomplete annotations. The 567 questions were selected as answer-usable; that does **not** claim that every video was manually verified. Compared with the corresponding original 698 rows, 120 answers, three question texts and three option sets changed. Use this release's answer as gold.

## Download

The package is published under `single_video_567/` in the public dataset repository `taryya/RunningBench`. No access request or login is needed:

```bash
hf download taryya/RunningBench --repo-type dataset \
  --include 'single_video_567/*' --local-dir ./RunningBench
cd RunningBench/single_video_567
sha256sum -c <(awk -F, 'NR>1 {print $4 "  " $2}' data/video_sha256.csv)
```

If the upload uses a dedicated dataset repository, replace the repository ID and omit the `single_video_567/` prefix. On Windows, verify hashes with Python or another SHA-256 utility. The upload is complete only when `data/video_sha256.csv` lists 567 files and all 567 `videos/*.mp4` are present.

## Evaluation rule

Use the same single-video protocol for every model and thinking configuration:

1. Supply one clip per question. Clips are 60-second segments, except 27 that end early at the end of the source recording (17–58 s). Extract 64 JPEG frames in temporal order with `ffmpeg` at `fps=64/duration`, `scale='min(480,iw)':-2`, quality 3, so every clip gets 64 frames regardless of length. The 480 limit applies to width; portrait height can exceed 480.
2. Use the fixed single-video prompt and deterministic option shuffle in `code/unified_eval.py`, temperature 0, and the same 567 IDs for every full run. Preserve the mapping from shuffled letters back to source options. The single-video prompt says the frames come from one clip of at most 60 seconds; it does not reuse multi-video wording about multiple clips.
3. Parse the answer strictly: use the last JSON object that has an `"answer"` key in the final (post-reasoning) text. There is no free-text letter fallback, and a letter outside the shown options is an error. `exact` counts only an identical set; `overlap` is correct-option recall; `wellformed` checks the required answer count. **The primary score uses all 567 questions as the denominator**: failed (truncated, unparseable, missing trace) and missing questions count as wrong. `score` also prints completed-only accuracy for diagnosis, plus `completed/failed/missing`. Do not mix a partial run with a full-run score.
4. Give each model/configuration its own run ID and result JSONL. Record the exact checkpoint or provider model ID, backend version, generation budget, mode, corpus hash, prompt hash, frame settings and per-question output.
5. For thinking-enabled models, preserve the complete raw response, extracted reasoning trace, `finish_reason`, usage and **every retry attempt**. Increase the output-token budget when `finish_reason=length`. A final response that still truncates, lacks a final answer, or lacks the expected visible reasoning is a failed question and counts as wrong. Do not silently parse a partial answer. The included Qwen3-VL Thinking vLLM run uses `--reasoning-parser qwen3`. Without a reasoning parser, the runner extracts inline `<think>…</think>` traces, including templates that pre-fill `<think>` so only `</think>` appears in the output.
6. Models with a real `enable_thinking` switch should run the same checkpoint twice, explicitly on and off. Fixed Instruct and fixed Thinking checkpoints are distinct model variants. Historical base result files do not prove thinking was off.

The multi-video result inventory has 26 base rows and 11 thinking-extension rows. Four thinking rows are fixed Thinking checkpoints already present in the base list; seven are extra thinking-on runs for switchable checkpoints. For the single-video suite, use **34 distinct configurations**: the 26 base model entries (set the seven switchable checkpoints explicitly to thinking off), those same seven checkpoints explicitly thinking on, and the separate Qwen3-VL-8B-Thinking checkpoint alongside Qwen3-VL-8B-Instruct. This gives 34 × 567 = **19,278 question-model evaluations**. Fixed Thinking checkpoints are run once. These are planned configuration slots; unavailable weights or APIs can prevent individual runs.

Match the multi-video harness settings that apply across video counts: deterministic per-question option shuffle, temperature 0, JSON answer format, exact/overlap/wellformed scoring, and identical frame pipeline. The single-video prompt is intentionally adapted to refer to one clip of at most 60 seconds. Use 64 total frames per question; allocate by clip duration with at least 4 frames per clip; `ffmpeg fps=k/duration,scale='min(480,iw)':-2 -q:v 3`. Each question has one clip, so it gets all 64 frames. Keep the original generation budgets for comparability: `--max-tokens 2048` for non-thinking and `--max-tokens 16384` for thinking, both with `--max-tokens-cap 32768`. If a response is explicitly truncated (`finish_reason=length`), the runner doubles the budget up to the cap to recover a complete answer; preserve both attempts and flag that recovery in the result. Thinking results retain the full raw message and extracted reasoning trace.

The fixed single-video prompt template is:

```text
Answer this multiple-choice question about egocentric walking/running footage.
The frames below are sampled in chronological order from the single video clip (at most 60 seconds long) associated with this question.

Watch the video frames before deciding. Base your answer on visual evidence from this video. Do not answer from the wording of the options alone.

QUESTION: {question}

OPTIONS:
{options}

Select exactly {n} option{plural}.
Return ONLY JSON: {"answer": [{example}]}
```

Example with an OpenAI-compatible service and the included runner:

```bash
python code/unified_eval.py inventory --dataset single_567_human \
  --root "$PWD" --single-clips "$PWD/videos" --work-dir "$PWD/work"
# Non-thinking model / thinking off
python code/unified_eval.py run --dataset single_567_human \
  --root "$PWD" --single-clips "$PWD/videos" --work-dir "$PWD/work" \
  --run-id MODEL-nothink --model EXACT_MODEL_ID --thinking false \
  --base-url http://127.0.0.1:8000/v1 --max-tokens 2048 --max-tokens-cap 32768
# Thinking model / thinking on
python code/unified_eval.py run --dataset single_567_human \
  --root "$PWD" --single-clips "$PWD/videos" --work-dir "$PWD/work" \
  --run-id MODEL-think --model EXACT_MODEL_ID --thinking true \
  --base-url http://127.0.0.1:8000/v1 --max-tokens 16384 --max-tokens-cap 32768
python code/unified_eval.py score --dataset single_567_human \
  --root "$PWD" --work-dir "$PWD/work" --run-id MODEL-think
python code/unified_eval.py compare --dataset single_567_human \
  --root "$PWD" --work-dir "$PWD/work"
```

For fixed Instruct or fixed Thinking checkpoints, use `--thinking auto` with the matching budget above. Re-running the same command resumes and retries only failed questions. `score` reports the primary `exact=N/567` over all questions, with the completed-only figure in brackets.

For switchable models set `--thinking true` or `--thinking false` explicitly. The backend must support `chat_template_kwargs.enable_thinking`; otherwise use a model-specific adapter that records the actual mode and preserves the trace. The bundled Floodgate adapter cannot verify `finish_reason` and is not suitable for the strict thinking/truncation rule without an extension.

## Release checklist

From the Hala server, the prepared upload folder is `~/datasets/RunningBench/single_video_567_hf_upload/` (567 video files, about 8.50 GB of video bytes). After Hugging Face authentication by a repository maintainer, upload **all** its contents to `single_video_567/` in the dataset repository:

```bash
hf auth login
hf upload taryya/RunningBench \
  ~/datasets/RunningBench/single_video_567_hf_upload single_video_567 \
  --repo-type dataset --commit-message "Add 567-question single-video evaluation package and clips"
```

Do not upload the private server path as a string in place of the video bytes. Verify the published repository by downloading it to a clean directory and checking video count, hashes and `inventory` output (`questions=567`, `media=567/567`).
