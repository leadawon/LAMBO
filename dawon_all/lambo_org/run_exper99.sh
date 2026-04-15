#!/usr/bin/env bash
# lambo_org — 99-sample Loong SET1 balanced retrieval-baseline experiment.
#
# Mirrors dawonv7/run_exper99.sh: build the deterministic balanced subset,
# then run inference restricted via --selected_indices_path.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LAMBO_ORG_PYTHON:-${LAMBO_DAWON_PYTHON:-/workspace/venvs/ragteamvenv/bin/python}}"
INPUT_PATH="${LAMBO_ORG_INPUT_PATH:-${LAMBO_DAWON_INPUT_PATH:-/workspace/StructRAG/loong/Loong/data/loong_process.jsonl}}"
DATA_DIR="${LAMBO_ORG_EXPER99_DATA_DIR:-${SCRIPT_DIR}/data}"
SUBSET_PATH="${LAMBO_ORG_EXPER99_SUBSET_PATH:-${DATA_DIR}/loong_set1_balanced99.jsonl}"
INDICES_PATH="${LAMBO_ORG_EXPER99_INDICES_PATH:-${DATA_DIR}/loong_set1_balanced99_indices.json}"
MANIFEST_PATH="${LAMBO_ORG_EXPER99_MANIFEST_PATH:-${DATA_DIR}/loong_set1_balanced99_manifest.json}"
OUTPUT_DIR="${LAMBO_ORG_OUTPUT_DIR:-${SCRIPT_DIR}/logs/lambo_org_set1_exper99}"
EMBEDDING_MODEL="${LAMBO_ORG_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
TOP_K_PER_DOC="${LAMBO_ORG_TOP_K_PER_DOC:-6}"
TOP_K_GLOBAL="${LAMBO_ORG_TOP_K_GLOBAL:-24}"
RETRIEVAL_SCOPE="${LAMBO_ORG_RETRIEVAL_SCOPE:-per_document}"
BACKEND="${LAMBO_ORG_BACKEND:-local}"
HELP_ONLY=0

for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    HELP_ONLY=1
  fi
done

if [[ "${HELP_ONLY}" -eq 1 ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_infer.py" "$@"
fi

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_exper99_subset.py" \
  --input "${INPUT_PATH}" \
  --subset-output "${SUBSET_PATH}" \
  --indices-output "${INDICES_PATH}" \
  --manifest-output "${MANIFEST_PATH}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_infer.py" \
  --input_path "${INPUT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --selected_indices_path "${INDICES_PATH}" \
  --embedding_model "${EMBEDDING_MODEL}" \
  --top_k_per_doc "${TOP_K_PER_DOC}" \
  --top_k_global "${TOP_K_GLOBAL}" \
  --retrieval_scope "${RETRIEVAL_SCOPE}" \
  --backend "${BACKEND}" \
  "$@"
