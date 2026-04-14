from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_TARGET_ROOT = Path("/workspace/qwen")
DEFAULT_MODEL_NAME = "Qwen2.5-32B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare /workspace/qwen for the local Qwen 32B model."
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help="Root directory that should contain the Qwen model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model directory name under the target root.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional existing model directory to reuse instead of downloading again.",
    )
    return parser.parse_args()


def resolve_source_dir(explicit_source: Path | None, model_name: str) -> Path | None:
    if explicit_source is not None:
        return explicit_source

    env_source = (
        os.environ.get("LAMBO_DAWONV4_EXISTING_MODEL_SOURCE", "").strip()
        or os.environ.get("LAMBO_DAWON_EXISTING_MODEL_SOURCE", "").strip()
    )
    if env_source:
        return Path(env_source).expanduser()

    candidates = [
        Path("/workspace/StructRAG/model") / model_name,
        Path("/workspace/meta-cognitive-RAG/models") / model_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_qwen_workspace(*, target_root: Path, model_name: str, source_dir: Path | None) -> dict[str, str]:
    target_root.mkdir(parents=True, exist_ok=True)
    target_dir = target_root / model_name

    if target_dir.exists():
        return {
            "status": "ready",
            "target_dir": str(target_dir),
            "mode": "existing",
        }

    if source_dir is None or not source_dir.exists():
        raise FileNotFoundError(
            "Could not find an existing Qwen 32B directory to reuse. "
            "Set --source, LAMBO_DAWONV4_EXISTING_MODEL_SOURCE, or "
            "LAMBO_DAWON_EXISTING_MODEL_SOURCE."
        )

    target_dir.symlink_to(source_dir, target_is_directory=True)
    return {
        "status": "ready",
        "target_dir": str(target_dir),
        "mode": "symlink",
        "source_dir": str(source_dir),
    }


def main() -> None:
    args = parse_args()
    source_dir = resolve_source_dir(args.source, args.model_name)
    result = ensure_qwen_workspace(
        target_root=args.target_root,
        model_name=args.model_name,
        source_dir=source_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
