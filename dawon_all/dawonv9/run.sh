#!/usr/bin/env bash
# dawonv9 runner — v8 architecture + Phase-2 cross-doc citation injection.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LAMBO_DAWONV9_PYTHON:-/workspace/venvs/ragteamvenv/bin/python}"
OUTPUT_DIR="${LAMBO_DAWONV9_OUTPUT_DIR:-${SCRIPT_DIR}/logs/dawonv9_set1_10}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_set1.py" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
