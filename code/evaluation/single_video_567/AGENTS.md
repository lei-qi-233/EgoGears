# Agent runbook: RunningBench single-video 567 evaluation

You are an AI coding agent asked to run this benchmark. Follow this file top to bottom. `README.md` gives the
background; this file is the operational procedure. Everything you need is in this folder. You do not need the
original authors' servers.

## Goal

Evaluate the 34 model configurations in `code/configs_34.csv` on the 567 questions in
`data/single_567_human_eval.jsonl` (one video per question, in `videos/`). Return one result JSONL per configuration
plus a summary table (see "Deliverables").

## Rules (do not break these, even to make a run succeed)

1. Do **not** edit `data/`, `videos/`, the prompt, the option shuffle, the parser or the scoring in
   `code/unified_eval.py`. If something seems wrong with them, stop and report it; do not work around it.
2. Use the `config_id` from `configs_34.csv` as the run ID, and keep its `model_id`, `thinking` and `max_tokens`
   values. Use one run ID per configuration and never reuse one for a different setting. The runner refuses a
   changed config under an existing run ID; that is intended.
3. Temperature is fixed at 0 in the runner. Do not add sampling parameters.
4. Failed questions count as wrong. Never delete failed records, and never drop questions to raise a score. The
   only allowed retry is re-running the same command, which retries only failed questions.
5. You may change serving details that do not affect answers: `TP`, `MAX_MODEL_LEN`, `VLLM_EXTRA_ARGS` (for
   example `--gpu-memory-utilization`, `--max-num-seqs`), `WORKERS`, `RPS`, `PORT`. Record every change you make.
6. If a configuration cannot run (weights unavailable, architecture unsupported by your vLLM, no API access, not
   enough GPUs), skip it and record why. Do not substitute a different checkpoint.

## Step 0: environment

- Linux, NVIDIA GPUs (80 GB class; `suggested_tp` in the CSV assumes 80 GB per GPU), Python ≥ 3.9, `ffmpeg`
  and `ffprobe` on `PATH`, `curl`.
- `pip install -U vllm huggingface_hub`. `code/unified_eval.py` itself uses only the standard library.
- Some checkpoints need recent vLLM/transformers (Qwen3.5, Gemma 4, GLM-4.6V). If one fails to load, try upgrading
  vLLM in a separate environment before skipping it.
- `google/gemma-3n-E4B-it` is gated on Hugging Face. The user must accept its license and log in with
  `hf auth login`. Do not enter tokens yourself; ask the user.
- Gemini configurations need a Google AI API key from the user (`API_KEY`), used through the OpenAI-compatible
  endpoint.

## Step 1: download and verify

```bash
hf download taryya/RunningBench --repo-type dataset --include 'single_video_567/*' --local-dir ./RunningBench
cd RunningBench/single_video_567
sha256sum -c <(awk -F, 'NR>1 {print $4 "  " $2}' data/video_sha256.csv) | grep -vc ': OK$'   # expect 0
python3 code/unified_eval.py inventory --dataset single_567_human --root "$PWD" --single-clips "$PWD/videos"
```

Expected inventory output: `questions=567`, `answer_count={4: 46, 1: 325, 2: 84, 3: 107, 5: 5}`, `media=567/567
missing=[]`. Stop if anything differs.

Optional: pre-extract frames once so later runs are faster (CPU only, a few minutes):
`python3 code/unified_eval.py prepare --dataset single_567_human --root "$PWD" --single-clips "$PWD/videos" --work-dir "$PWD/work"`.

## Step 2: for each configuration, smoke test first, then run in full

All commands run from the package root. Pick a `config_id` from `code/configs_34.csv`.

```bash
LIMIT=3 bash code/run_config.sh qwen3-vl-8b-instruct     # smoke test: starts vLLM, runs 3 questions
```

Check the smoke output before the full run. Inspect `work/runs/<config_id>/single_567_human.jsonl`:

| Check | Expected |
|---|---|
| `error` | `null` on all 3 rows. Otherwise read the message; see Troubleshooting |
| `finish_reason` | `"stop"` |
| `pred` | a list of letters; `n_pred` usually equals `n_select` |
| `thinking=true` configs, and fixed Thinking checkpoints | `thinking_trace_present: true`, and `reasoning_content` holds real reasoning |
| `thinking=false` configs | `reasoning_content` is empty/null. If a trace still appears, the backend ignored `enable_thinking`. Record this and report it; do not publish the run as "thinking off" |
| `n_frames` | 64 |

Then run the full set. It is the same command without `LIMIT` and reuses the smoke results:

```bash
bash code/run_config.sh qwen3-vl-8b-instruct
bash code/run_config.sh qwen3-vl-8b-instruct   # run once more if the first pass had transient failures
```

The `score` step at the end prints the primary metric, for example
`overall: exact=N/567 (x%) [completed-only ...] overlap=... wellformed=... failed=F missing=0`.
`missing` must be 0 for a full run.

Configurations that share a checkpoint (`*-nothink` / `*-think`) can reuse one server: start vLLM yourself, then run
both with `BASE_URL=http://127.0.0.1:8000/v1 bash code/run_config.sh <config_id>`.

For Gemini (`backend=provider`):

```bash
BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai API_KEY=<from user> RPS=0.5 WORKERS=2 \
  bash code/run_config.sh gemini-2.5-pro
```

Suggested order: small models first (`qwen3-vl-8b-instruct`, `qwen3-vl-8b-thinking`, `qwen2.5-vl-7b-instruct`,
`internvl3_5-8b`, ...), then larger ones as GPUs allow.

## Step 3: deliverables

```bash
python3 code/unified_eval.py compare --dataset single_567_human --root "$PWD" --work-dir "$PWD/work" > results_summary.tsv
tar czf runningbench567_results.tgz results_summary.tsv work/runs $([ -d work/logs ] && echo work/logs)
```

Send back `runningbench567_results.tgz` along with a short note that lists:

- for each config: completed, `exact=N/567`, and any serving changes (TP, MAX_MODEL_LEN, extra args);
- skipped configs and why;
- `vllm --version`, `python -c "import transformers; print(transformers.__version__)"`, GPU model and count;
- anything unexpected, such as a thinking switch that was ignored, many `finish_reason=length` failures, or
  parse failures.

Do not include `work/cache` (extracted frames). It is large and can be regenerated.

## Troubleshooting

| Symptom | Action |
|---|---|
| HTTP 400 mentioning maximum context length | Raise `MAX_MODEL_LEN` up to the model's limit. Where the model's limit is smaller (for example Gemma-3n 32K), use that limit and accept the resulting truncation failures |
| `Output truncated at max_tokens_cap=32768` | Expected occasionally for thinking models. It counts as wrong; leave it |
| `No parseable final answer: ...` | The model did not return the required JSON. It counts as wrong; do not loosen the parser |
| `Thinking trace missing from model response` | For Thinking checkpoints, try the model's vLLM `--reasoning-parser` via `VLLM_EXTRA_ARGS` in a **new** run ID suffix (e.g. `<config_id>-rp`), and report both |
| CUDA OOM at startup | Increase `TP`, or add `VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.95 --max-num-seqs 4"` |
| Model architecture not supported | Upgrade vLLM; if it still fails, skip and report |
| `Run configuration changed; choose another --run-id` | You changed `model`/`thinking`/`max_tokens`. Revert to the CSV values |

`code/run_qwen3vl8b_batch.sh` and `code/build_single_567_eval.py` are the original authors' Slurm/data-building
scripts for their own cluster. You do not need them.
