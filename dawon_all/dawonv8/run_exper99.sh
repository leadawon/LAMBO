#!/usr/bin/env bash
# dawonv8 — 99-sample Loong SET1 balanced experiment.
#
# Mirrors dawonv7/run_exper99.sh: build the deterministic balanced subset,
# then hand off to dawonv8/run.sh restricting processing via
# --selected_indices_path.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LAMBO_DAWONV8_PYTHON:-${LAMBO_DAWON_PYTHON:-/workspace/venvs/ragteamvenv/bin/python}}"
INPUT_PATH="${LAMBO_DAWONV8_INPUT_PATH:-${LAMBO_DAWON_INPUT_PATH:-/workspace/StructRAG/loong/Loong/data/loong_process.jsonl}}"
DATA_DIR="${LAMBO_DAWONV8_EXPER99_DATA_DIR:-${SCRIPT_DIR}/data}"
SUBSET_PATH="${LAMBO_DAWONV8_EXPER99_SUBSET_PATH:-${DATA_DIR}/loong_set1_balanced99.jsonl}"
INDICES_PATH="${LAMBO_DAWONV8_EXPER99_INDICES_PATH:-${DATA_DIR}/loong_set1_balanced99_indices.json}"
MANIFEST_PATH="${LAMBO_DAWONV8_EXPER99_MANIFEST_PATH:-${DATA_DIR}/loong_set1_balanced99_manifest.json}"
OUTPUT_DIR="${LAMBO_DAWONV8_OUTPUT_DIR:-${SCRIPT_DIR}/logs/dawonv8_set1_exper99}"
HELP_ONLY=0

for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    HELP_ONLY=1
  fi
done

if [[ "${HELP_ONLY}" -eq 1 ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_set1.py" "$@"
fi

mkdir -p "${DATA_DIR}" "$(dirname "${OUTPUT_DIR}")"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_exper99_subset.py" \
  --input "${INPUT_PATH}" \
  --subset-output "${SUBSET_PATH}" \
  --indices-output "${INDICES_PATH}" \
  --manifest-output "${MANIFEST_PATH}"

export LAMBO_DAWONV8_INPUT_PATH="${INPUT_PATH}"
export LAMBO_DAWONV8_OUTPUT_DIR="${OUTPUT_DIR}"

exec bash "${SCRIPT_DIR}/run.sh" \
  --input_path "${INPUT_PATH}" \
  --selected_indices_path "${INDICES_PATH}" \
  "$@"
