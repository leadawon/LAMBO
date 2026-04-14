from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .common import read_jsonl, write_json


SELECTED_SET1_INDICES = [914, 725, 1065, 801, 1280, 469, 63, 1504, 596, 32]


@dataclass
class ManifestItem:
    selected_index: int
    record_id: str
    set_id: int
    record_type: str
    level: int
    language: str
    question: str
    sample_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_records(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def build_set1_manifest(records: Sequence[Dict[str, Any]]) -> List[ManifestItem]:
    manifest: List[ManifestItem] = []
    for index in SELECTED_SET1_INDICES:
        record = records[index]
        manifest.append(
            ManifestItem(
                selected_index=index,
                record_id=str(record.get("id", "unknown")),
                set_id=int(record.get("set", 0) or 0),
                record_type=str(record.get("type", "unknown")),
                level=int(record.get("level", 0) or 0),
                language=str(record.get("language", "unknown")),
                question=str(record.get("question", "")).strip(),
                sample_id=f"{record.get('type', 'unknown')}_level{record.get('level', 0)}_{index}",
            )
        )
    return manifest



def build_manifest_for_indices(records: Sequence[Dict[str, Any]], indices: Iterable[int]) -> List[ManifestItem]:
    manifest: List[ManifestItem] = []
    seen = set()
    for raw_index in indices:
        index = int(raw_index)
        if index in seen:
            continue
        if index < 0 or index >= len(records):
            raise IndexError(f"selected index out of range: {index}")
        seen.add(index)
        record = records[index]
        manifest.append(
            ManifestItem(
                selected_index=index,
                record_id=str(record.get("id", "unknown")),
                set_id=int(record.get("set", 0) or 0),
                record_type=str(record.get("type", "unknown")),
                level=int(record.get("level", 0) or 0),
                language=str(record.get("language", "unknown")),
                question=str(record.get("question", "")).strip(),
                sample_id=f"{record.get('type', 'unknown')}_level{record.get('level', 0)}_{index}",
            )
        )
    return manifest

def save_manifest(items: Sequence[ManifestItem], path: Path) -> Path:
    return write_json(path, [item.to_dict() for item in items])
