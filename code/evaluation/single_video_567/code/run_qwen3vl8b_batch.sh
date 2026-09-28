#!/usr/bin/env bash
# Submit with: sbatch run_qwen3vl8b_batch.sh
# Runs the Instruct and Thinking 8B checkpoints on human-curated single-video QA only.
#SBATCH --job-name=rb-qwen3vl8b-pilot
#SBATCH --partition=batch
#SBATCH --nodelist=hala
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yuedong_tan/datasets/RunningBench/unified_eval/work/logs/%x-%j.out

set -euo pipefail

ROOT=/home/yuedong_tan/datasets/RunningBench/unified_eval
PY="$ROOT/.venv_clean/bin/python"
VLLM="$ROOT/.venv_clean/bin/vllm"
PORT=$((18000 + SLURM_JOB_ID % 10000))
EVAL_DATASET=single_567_human
EVAL_SKIP="${RB_SKIP:-0}"
EVAL_LIMIT="${RB_LIMIT:-2}"
RUN_TAG_SUFFIX="${RB_RUN_TAG_SUFFIX:-human567-singleprompt-pilot}"
mkdir -p "$ROOT/work/logs"
export HF_HOME="${HF_HOME:-/home/yuedong_tan/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

serve_and_eval() {
    local model="$1"
    local run_id="$2"
    local max_tokens="$3"
    local max_tokens_cap="$4"
    local log="$ROOT/work/logs/${run_id}-${SLURM_JOB_ID}-server.log"
    local parser_args=()
    if [[ "$model" == *Thinking ]]; then
        parser_args=(--reasoning-parser qwen3)
    fi
    echo "Starting $model on $(hostname), port $PORT"
    "$VLLM" serve "$model" \
        --host 127.0.0.1 --port "$PORT" \
        --dtype bfloat16 --max-model-len 49152 \
        --limit-mm-per-prompt '{"image":64,"video":0}' \
        --max-num-seqs 1 --gpu-memory-utilization 0.90 \
        "${parser_args[@]}" >"$log" 2>&1 &
    SERVER_PID=$!
    trap 'kill "${SERVER_PID:-}" 2>/dev/null || true' EXIT
    local ready=0
    for ((i=0; i<120; i++)); do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server exited; see $log"
            tail -80 "$log"
            return 1
        fi
        sleep 15
    done
    if [[ "$ready" != 1 ]]; then
        echo "Server readiness timed out; see $log"
        tail -80 "$log"
        return 1
    fi
    "$PY" "$ROOT/unified_eval.py" run --dataset "$EVAL_DATASET" \
        --skip "$EVAL_SKIP" --limit "$EVAL_LIMIT" \
        --run-id "$run_id" --model "$model" --thinking auto \
        --max-tokens "$max_tokens" --max-tokens-cap "$max_tokens_cap" \
        --workers 6 --rps 0.45 \
        --base-url "http://127.0.0.1:$PORT/v1"
    "$PY" "$ROOT/unified_eval.py" score --dataset "$EVAL_DATASET" --run-id "$run_id"
    kill "$SERVER_PID" 2>/dev/null || true
    for ((i=0; i<15; i++)); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
    trap - EXIT
}

"$PY" -m pip show vllm transformers torch | grep -E 'Name:|Version:'
serve_and_eval Qwen/Qwen3-VL-8B-Instruct "qwen3vl8b-instruct-$RUN_TAG_SUFFIX" 2048 32768
serve_and_eval Qwen/Qwen3-VL-8B-Thinking "qwen3vl8b-thinking-$RUN_TAG_SUFFIX" 16384 32768
"$PY" "$ROOT/unified_eval.py" compare --dataset single_567_human
