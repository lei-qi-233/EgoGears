#!/usr/bin/env bash
# Run one configuration from code/configs_34.csv end to end: start vLLM (unless BASE_URL is set),
# evaluate the 567 questions, and print the score. Re-running the same CONFIG_ID resumes and
# retries only failed questions.
#
# Usage: bash code/run_config.sh CONFIG_ID
# Env:   LIMIT=N           evaluate only the first N questions (smoke test); default 0 = all 567
#        BASE_URL=...      use an existing OpenAI-compatible server (required for provider models)
#        API_KEY=...       bearer token for BASE_URL (default: $OPENAI_API_KEY or EMPTY)
#        TP=N              tensor-parallel size (default: suggested_tp from the CSV)
#        MAX_MODEL_LEN=N   vLLM context length (default 65536)
#        VLLM_EXTRA_ARGS   extra flags appended to `vllm serve`
#        PORT, WORKERS (default 4), RPS (default 2), WORK_DIR (default ./work), PYTHON, VLLM
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
CONFIG_ID=${1:?usage: bash code/run_config.sh CONFIG_ID   (IDs are in code/configs_34.csv)}
PYTHON=${PYTHON:-python3}
VLLM=${VLLM:-vllm}
WORK_DIR=${WORK_DIR:-$HERE/work}
LIMIT=${LIMIT:-0}

FIELDS=$("$PYTHON" - "$HERE/code/configs_34.csv" "$CONFIG_ID" <<'EOF'
import csv, sys
rows = {r["config_id"]: r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8"))}
r = rows.get(sys.argv[2])
if r is None:
    sys.exit(f"Unknown config_id {sys.argv[2]!r}; valid IDs: {', '.join(rows)}")
print(r["model_id"], r["backend"], r["thinking"], r["max_tokens"], r["reasoning_parser"] or "-", r["suggested_tp"] or "1")
EOF
)
read -r MODEL BACKEND THINKING MAX_TOKENS PARSER TP_DEFAULT <<<"$FIELDS"
echo "CONFIG $CONFIG_ID model=$MODEL backend=$BACKEND thinking=$THINKING max_tokens=$MAX_TOKENS limit=$LIMIT"

SERVER_PID=
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [[ -z "${BASE_URL:-}" ]]; then
    if [[ "$BACKEND" == provider ]]; then
        echo "Provider model: set BASE_URL (e.g. https://generativelanguage.googleapis.com/v1beta/openai) and API_KEY" >&2
        exit 2
    fi
    PORT=${PORT:-8000}
    mkdir -p "$WORK_DIR/logs"
    LOG="$WORK_DIR/logs/$CONFIG_ID-server.log"
    ARGS=(serve "$MODEL" --host 127.0.0.1 --port "$PORT" --trust-remote-code --dtype bfloat16
          --max-model-len "${MAX_MODEL_LEN:-65536}" --limit-mm-per-prompt '{"image":64,"video":0}'
          --tensor-parallel-size "${TP:-$TP_DEFAULT}")
    [[ "$PARSER" != "-" ]] && ARGS+=(--reasoning-parser "$PARSER")
    # shellcheck disable=SC2206
    ARGS+=(${VLLM_EXTRA_ARGS:-})
    echo "Starting: $VLLM ${ARGS[*]}  (log: $LOG)"
    "$VLLM" "${ARGS[@]}" >"$LOG" 2>&1 &
    SERVER_PID=$!
    for ((i = 0; i < 240; i++)); do
        curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM exited during startup; last log lines:" >&2; tail -60 "$LOG" >&2; exit 3
        fi
        sleep 15
    done
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "vLLM not ready after 60 min; see $LOG" >&2; exit 3; }
    BASE_URL="http://127.0.0.1:$PORT/v1"
fi

KEY_ARGS=()
[[ -n "${API_KEY:-}" ]] && KEY_ARGS=(--api-key "$API_KEY")
COMMON=(--dataset single_567_human --root "$HERE" --work-dir "$WORK_DIR" --run-id "$CONFIG_ID")
"$PYTHON" "$HERE/code/unified_eval.py" run "${COMMON[@]}" --single-clips "$HERE/videos" \
    --model "$MODEL" --thinking "$THINKING" --max-tokens "$MAX_TOKENS" --max-tokens-cap 32768 \
    --workers "${WORKERS:-4}" --rps "${RPS:-2}" --limit "$LIMIT" --base-url "$BASE_URL" ${KEY_ARGS[@]+"${KEY_ARGS[@]}"}
"$PYTHON" "$HERE/code/unified_eval.py" score "${COMMON[@]}"
