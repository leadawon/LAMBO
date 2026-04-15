#!/usr/bin/env bash
# dawonv8 runner — lambo v2 architecture + v5/v7 enrichment.
#
# The lambo backend uses LocalTransformersBackend directly (no separate
# vLLM server), so this script just invokes run_set1.py. If you need the
# vLLM server setup from dawonv7, copy dawonv7/run.sh and point it at
# dawonv8/run_set1.py.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LAMBO_DAWONV8_PYTHON:-/workspace/venvs/ragteamvenv/bin/python}"
OUTPUT_DIR="${LAMBO_DAWONV8_OUTPUT_DIR:-${SCRIPT_DIR}/logs/dawonv8_set1_10}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_set1.py" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
