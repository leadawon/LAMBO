#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LAMBO_DAWONV4_PYTHON:-${LAMBO_DAWON_PYTHON:-/workspace/venvs/ragteamvenv/bin/python}}"
OUTPUT_DIR="${LAMBO_DAWONV4_OUTPUT_DIR:-${SCRIPT_DIR}/logs/lambo_set1_exper99}"
INPUT_PATH="${LAMBO_DAWONV4_INPUT_PATH:-${LAMBO_DAWON_INPUT_PATH:-/workspace/StructRAG/loong/Loong/data/loong_process.jsonl}}"
DATA_DIR="${LAMBO_DAWONV4_EXPER99_DATA_DIR:-${SCRIPT_DIR}/data}"
SUBSET_PATH="${LAMBO_DAWONV4_EXPER99_SUBSET_PATH:-${DATA_DIR}/loong_set1_balanced99.jsonl}"
INDICES_PATH="${LAMBO_DAWONV4_EXPER99_INDICES_PATH:-${DATA_DIR}/loong_set1_balanced99_indices.json}"
MANIFEST_PATH="${LAMBO_DAWONV4_EXPER99_MANIFEST_PATH:-${DATA_DIR}/loong_set1_balanced99_manifest.json}"
SERVER_LOG="${LAMBO_DAWONV4_SERVER_LOG:-${SCRIPT_DIR}/logs/backend_server_exper99.log}"
HELP_ONLY=0

for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    HELP_ONLY=1
  fi
done

if [[ "${HELP_ONLY}" -eq 1 ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_set1_10.py" "$@"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_exper99_subset.py" \
  --input "${INPUT_PATH}" \
  --subset-output "${SUBSET_PATH}" \
  --indices-output "${INDICES_PATH}" \
  --manifest-output "${MANIFEST_PATH}"

export LAMBO_DAWONV4_INPUT_PATH="${INPUT_PATH}"
export LAMBO_DAWONV4_OUTPUT_DIR="${OUTPUT_DIR}"
export LAMBO_DAWONV4_SERVER_LOG="${SERVER_LOG}"

# Pass the original full dataset as --input_path and restrict processing to the
# 99-sample subset via --selected_indices_path (which holds the original indices).
exec bash "${SCRIPT_DIR}/run.sh" \
  --input_path "${INPUT_PATH}" \
  --selected_indices_path "${INDICES_PATH}" \
  "$@"
