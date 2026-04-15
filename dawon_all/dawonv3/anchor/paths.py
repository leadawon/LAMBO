from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


DAWON_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DAWON_ROOT.parent


def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_path(
    *,
    env_name: str,
    candidates: Iterable[Path],
    label: str,
    strict: bool,
) -> Optional[Path]:
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        path = Path(env_value).expanduser()
        if strict and not path.exists():
            raise FileNotFoundError(f"{label} not found via {env_name}: {path}")
        return path

    resolved = _first_existing(candidates)
    if resolved is not None:
        return resolved

    if not strict:
        for candidate in candidates:
            return candidate
        return None

    tried = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve {label}. Set {env_name} or create one of:\n{tried}"
    )


def resolve_loong_process_path(strict: bool = True) -> Optional[Path]:
    return _resolve_path(
        env_name="LAMBO_DAWON_INPUT_PATH",
        candidates=[
            PROJECT_ROOT / "Loong" / "data" / "loong_process.jsonl",
            Path("/workspace/Plan_Search_RAG/Loong/data/loong_process.jsonl"),
            Path("/workspace/StructRAG/loong/Loong/data/loong_process.jsonl"),
        ],
        label="Loong processed input file",
        strict=strict,
    )


def resolve_loong_jsonl_path(strict: bool = True) -> Optional[Path]:
    return _resolve_path(
        env_name="LAMBO_DAWON_LOONG_JSONL",
        candidates=[
            PROJECT_ROOT / "Loong" / "data" / "loong.jsonl",
            Path("/workspace/Plan_Search_RAG/Loong/data/loong.jsonl"),
            Path("/workspace/StructRAG/loong/Loong/data/loong.jsonl"),
        ],
        label="Loong raw jsonl file",
        strict=strict,
    )


def resolve_loong_src(strict: bool = True) -> Optional[Path]:
    return _resolve_path(
        env_name="LAMBO_DAWON_LOONG_SRC",
        candidates=[
            PROJECT_ROOT / "Loong" / "src",
            Path("/workspace/Plan_Search_RAG/Loong/src"),
            Path("/workspace/StructRAG/loong/Loong/src"),
        ],
        label="Loong source directory",
        strict=strict,
    )


def resolve_loong_model_dir(strict: bool = True) -> Optional[Path]:
    return _resolve_path(
        env_name="LAMBO_DAWON_LOONG_MODEL_DIR",
        candidates=[
            PROJECT_ROOT / "Loong" / "config" / "models",
            Path("/workspace/Plan_Search_RAG/Loong/config/models"),
            Path("/workspace/StructRAG/loong/Loong/config/models"),
        ],
        label="Loong model config directory",
        strict=strict,
    )


def resolve_local_model_dir(strict: bool = True) -> Optional[Path]:
    return _resolve_path(
        env_name="LAMBO_DAWON_MODEL_DIR",
        candidates=[
            Path("/workspace/qwen/Qwen2.5-32B-Instruct"),
            Path("/workspace/qwen/qwen2.5-32b-instruct"),
            Path("/workspace/meta-cognitive-RAG/models/Qwen2.5-32B-Instruct"),
            PROJECT_ROOT / "meta-cognitive-RAG" / "models" / "Qwen2.5-32B-Instruct",
            Path("/workspace/StructRAG/model/Qwen2.5-32B-Instruct"),
        ],
        label="local Qwen model directory",
        strict=strict,
    )
