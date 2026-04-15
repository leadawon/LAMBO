#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LAMBO_DAWON_PYTHON:-/workspace/venvs/ragteamvenv/bin/python}"
OUTPUT_DIR="${LAMBO_DAWON_OUTPUT_DIR:-${SCRIPT_DIR}/logs/lambo_set1_10}"
SERVER_PYTHON="${LAMBO_DAWON_SERVER_PYTHON:-${PYTHON_BIN}}"
LLM_BACKEND="${LAMBO_DAWON_LLM_BACKEND:-server}"
SERVER_HOST="${LAMBO_DAWON_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${LAMBO_DAWON_SERVER_PORT:-1225}"
SERVER_MODEL_NAME="${LAMBO_DAWON_SERVER_MODEL_NAME:-Qwen}"
SERVER_LOG="${LAMBO_DAWON_SERVER_LOG:-${SCRIPT_DIR}/logs/backend_server.log}"
SERVER_MAX_MODEL_LEN="${LAMBO_DAWON_SERVER_MAX_MODEL_LEN:-32768}"
SERVER_GPU_MEMORY_UTILIZATION="${LAMBO_DAWON_SERVER_GPU_MEMORY_UTILIZATION:-0.88}"
SERVER_TP_SIZE="${LAMBO_DAWON_SERVER_TP_SIZE:-4}"
SERVER_MAX_NUM_SEQS="${LAMBO_DAWON_SERVER_MAX_NUM_SEQS:-1}"
SERVER_HEALTH_TIMEOUT="${LAMBO_DAWON_SERVER_HEALTH_TIMEOUT:-1800}"
SERVER_GUIDED_DECODING_BACKEND="${LAMBO_DAWON_SERVER_GUIDED_DECODING_BACKEND:-lm-format-enforcer}"
HELP_ONLY=0
SERVER_PID=""

for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    HELP_ONLY=1
  fi
done

if [[ "${HELP_ONLY}" -eq 1 ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_set1_10.py" "$@"
fi

cleanup() {
  local status=$?
  if [[ -n "${SERVER_PID}" ]]; then
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Stopping backend server (pid=${SERVER_PID})"
      kill -TERM -"${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
      sleep 5
      if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -KILL -"${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
      fi
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  exit "${status}"
}

wait_for_server() {
  "${PYTHON_BIN}" - <<'PY'
import os
import sys
import time
import requests

host = os.environ["LAMBO_DAWON_SERVER_HOST"]
port = os.environ["LAMBO_DAWON_SERVER_PORT"]
timeout_s = float(os.environ.get("LAMBO_DAWON_SERVER_HEALTH_TIMEOUT", "600"))
deadline = time.time() + timeout_s
url = f"http://{host}:{port}/health"
server_pid = os.environ.get("LAMBO_DAWON_SERVER_PID", "").strip()
last_error = ""

def server_exited(pid: str) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    stat_path = f"/proc/{pid}/stat"
    try:
        state = open(stat_path, encoding="utf-8").read().split()[2]
    except Exception:
        return False
    return state == "Z"

while time.time() < deadline:
    if server_pid and server_exited(server_pid):
        print(f"Backend server exited before becoming ready: pid={server_pid}", file=sys.stderr)
        sys.exit(1)
    try:
        response = requests.get(url, timeout=5)
        if response.ok:
            print(f"Backend server ready: {url}")
            sys.exit(0)
        last_error = f"status={response.status_code} body={response.text[:200]}"
    except Exception as exc:
        last_error = repr(exc)
    time.sleep(2)

print(
  f"Backend server did not become ready within {timeout_s:.0f}s: {last_error}. "
  "Large models may require a longer startup window; set LAMBO_DAWON_SERVER_HEALTH_TIMEOUT.",
  file=sys.stderr,
)
sys.exit(1)
PY
}

vllm_supports_arg() {
  local flag="$1"
  "${SERVER_PYTHON}" -m vllm.entrypoints.openai.api_server --help 2>&1 | grep -q -- "${flag}"
}

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_qwen_workspace.py"

export LAMBO_DAWON_MODEL_DIR="${LAMBO_DAWON_MODEL_DIR:-/workspace/qwen/Qwen2.5-32B-Instruct}"
export LAMBO_DAWON_LLM_BACKEND="${LLM_BACKEND}"
export LAMBO_DAWON_SERVER_HOST="${SERVER_HOST}"
export LAMBO_DAWON_SERVER_PORT="${SERVER_PORT}"
export LAMBO_DAWON_SERVER_MODEL_NAME="${SERVER_MODEL_NAME}"
export LAMBO_DAWON_SERVER_HEALTH_TIMEOUT="${SERVER_HEALTH_TIMEOUT}"
export LAMBO_DAWON_BASE_URL="${LAMBO_DAWON_BASE_URL:-http://${SERVER_HOST}:${SERVER_PORT}/v1}"

trap cleanup EXIT INT TERM

if [[ "${HELP_ONLY}" -eq 0 && "${LLM_BACKEND}" == "server" ]]; then
  mkdir -p "$(dirname "${SERVER_LOG}")"
  echo "Starting backend server on ${SERVER_HOST}:${SERVER_PORT}"
  echo "Waiting up to ${SERVER_HEALTH_TIMEOUT}s for backend health check"
  SERVER_ARGS=(
    -m vllm.entrypoints.openai.api_server
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    --served-model-name "${SERVER_MODEL_NAME}" \
    --model "${LAMBO_DAWON_MODEL_DIR}" \
    --tensor-parallel-size "${SERVER_TP_SIZE}" \
    --max-model-len "${SERVER_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${SERVER_GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${SERVER_MAX_NUM_SEQS}" \
    --enforce-eager \
    --disable-custom-all-reduce
  )
  if [[ -n "${SERVER_GUIDED_DECODING_BACKEND}" ]]; then
    if vllm_supports_arg "--guided-decoding-backend"; then
      SERVER_ARGS+=(--guided-decoding-backend "${SERVER_GUIDED_DECODING_BACKEND}")
    else
      echo "Skipping unsupported vLLM option: --guided-decoding-backend"
    fi
  fi
  setsid "${SERVER_PYTHON}" "${SERVER_ARGS[@]}" \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  export LAMBO_DAWON_SERVER_PID="${SERVER_PID}"
  if ! wait_for_server; then
    echo "Backend server failed to start. Last log lines:"
    tail -n 80 "${SERVER_LOG}" || true
    exit 1
  fi
fi

set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_set1_10.py" \
  --output_dir "${OUTPUT_DIR}" \
  --force \
  --skip_judge \
  "$@"
RUN_STATUS=$?
set -e

exit "${RUN_STATUS}"
