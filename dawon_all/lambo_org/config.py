"""Central configuration for the lambo_org retrieval baseline.

Exposes tunables that matter for a clean retrieval baseline:
- embedding model
- chunking policy
- top-k
- retrieval scope (global vs per-document)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LAMBO_ROOT = Path(__file__).resolve().parents[2]
DAWON_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = Path(__file__).resolve().parent

# Match dawonv7 / dawonv8 conventions.
DEFAULT_INPUT_PATH = Path("/workspace/StructRAG/loong/Loong/data/loong_process.jsonl")
DEFAULT_DATA_DIR = PKG_ROOT / "data"
DEFAULT_SUBSET_OUTPUT = DEFAULT_DATA_DIR / "loong_set1_balanced99.jsonl"
DEFAULT_INDICES_OUTPUT = DEFAULT_DATA_DIR / "loong_set1_balanced99_indices.json"
DEFAULT_MANIFEST_OUTPUT = DEFAULT_DATA_DIR / "loong_set1_balanced99_manifest.json"
DEFAULT_OUTPUT_DIR = PKG_ROOT / "logs" / "lambo_org_set1_exper99"


@dataclass
class LamboOrgConfig:
    # --- Retrieval baseline ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_per_doc: int = 6
    top_k_global: int = 24
    retrieval_scope: str = "per_document"  # "per_document" | "global"
    # --- Chunking ---
    max_units_per_doc: int = 220
    chunk_target_chars: int = 1200
    # --- IO ---
    input_path: Path = DEFAULT_INPUT_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    subset_path: Path = DEFAULT_SUBSET_OUTPUT
    indices_path: Path = DEFAULT_INDICES_OUTPUT
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT
    # --- Misc ---
    force: bool = False
    max_refine_rounds: int = 1  # retrieval baseline is single-pass
    backend: str = "local"
    embedding_cache_dir: Optional[Path] = None
