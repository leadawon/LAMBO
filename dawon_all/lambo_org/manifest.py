"""Manifest helpers for lambo_org. Mirrors dawonv7 conventions so the
99-example experiment can be driven by a ``selected_indices`` JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from lambo.common import read_jsonl, write_json
from lambo.manifest import ManifestItem, SELECTED_SET1_INDICES  # reuse schema


def load_records(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def _item(record: Dict[str, Any], index: int) -> ManifestItem:
    return ManifestItem(
        selected_index=index,
        record_id=str(record.get("id", "unknown")),
        set_id=int(record.get("set", 0) or 0),
        record_type=str(record.get("type", "unknown")),
        level=int(record.get("level", 0) or 0),
        language=str(record.get("language", "unknown")),
        question=str(record.get("question", "")).strip(),
        sample_id=f"{record.get('type', 'unknown')}_level{record.get('level', 0)}_{index}",
    )


def build_manifest_for_indices(
    records: Sequence[Dict[str, Any]], indices: Sequence[int]
) -> List[ManifestItem]:
    return [_item(records[i], i) for i in indices]


def build_set1_manifest(records: Sequence[Dict[str, Any]]) -> List[ManifestItem]:
    return build_manifest_for_indices(records, SELECTED_SET1_INDICES)


def parse_index_tokens(text: str) -> List[int]:
    return [int(token) for token in text.replace(",", " ").split()]


def load_selected_indices(selected: str, selected_path: str) -> List[int] | None:
    indices: List[int] = []
    if selected_path:
        text = Path(selected_path).read_text(encoding="utf-8").strip()
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                indices.extend(parse_index_tokens(text))
            else:
                if isinstance(payload, dict):
                    payload = payload.get("indices", [])
                if not isinstance(payload, list):
                    raise ValueError(
                        f"selected_indices_path must contain a JSON list or "
                        f"dict.indices: {selected_path}"
                    )
                indices.extend(int(v) for v in payload)
    if selected:
        indices.extend(parse_index_tokens(selected))
    return indices or None


def save_manifest(items: Sequence[ManifestItem], path: Path) -> Path:
    return write_json(path, [item.to_dict() for item in items])
