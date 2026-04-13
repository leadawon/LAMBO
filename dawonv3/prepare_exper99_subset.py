#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


DEFAULT_INPUT = Path("/workspace/StructRAG/loong/Loong/data/loong_process.jsonl")
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SUBSET_OUTPUT = DEFAULT_DATA_DIR / "loong_set1_balanced99.jsonl"
DEFAULT_INDICES_OUTPUT = DEFAULT_DATA_DIR / "loong_set1_balanced99_indices.json"
DEFAULT_MANIFEST_OUTPUT = DEFAULT_DATA_DIR / "loong_set1_balanced99_manifest.json"
DOMAIN_ALIASES = {
    "finance": "financial",
    "financial": "financial",
    "legal": "legal",
    "law": "legal",
    "paper": "paper",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a deterministic balanced SET1 99-sample Loong subset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--subset-output", type=Path, default=DEFAULT_SUBSET_OUTPUT)
    parser.add_argument("--indices-output", type=Path, default=DEFAULT_INDICES_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--set-id", type=int, default=1)
    parser.add_argument("--per-domain", type=int, default=33)
    parser.add_argument("--domains", type=str, default="paper,legal,financial")
    return parser.parse_args()


def normalize_domains(raw_domains: str) -> List[str]:
    domains = []
    for token in raw_domains.replace(",", " ").split():
        normalized = DOMAIN_ALIASES.get(token.strip().lower())
        if normalized is None:
            raise ValueError(f"Unsupported domain: {token}")
        domains.append(normalized)
    if not domains:
        raise ValueError("At least one domain is required")
    return domains


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if line:
                yield index, json.loads(line)


def main() -> None:
    args = parse_args()
    domains = normalize_domains(args.domains)

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for index, row in iter_jsonl(args.input):
        record_type = str(row.get("type", "")).lower()
        if row.get("set") != args.set_id or record_type not in domains:
            continue
        if len(buckets[record_type]) >= args.per_domain:
            continue
        subset_row = dict(row)
        subset_row["selected_index"] = index
        buckets[record_type].append(subset_row)
        if all(len(buckets[domain]) >= args.per_domain for domain in domains):
            break

    missing = {domain: args.per_domain - len(buckets[domain]) for domain in domains if len(buckets[domain]) < args.per_domain}
    if missing:
        raise SystemExit(f"Not enough SET{args.set_id} rows for balanced subset: {missing}")

    selected: List[Dict[str, Any]] = []
    for domain in domains:
        selected.extend(buckets[domain][: args.per_domain])

    counts = Counter(row.get("type") for row in selected)
    indices = [int(row["selected_index"]) for row in selected]
    manifest = {
        "source_path": str(args.input),
        "subset_output": str(args.subset_output),
        "indices_output": str(args.indices_output),
        "set_id": args.set_id,
        "per_domain": args.per_domain,
        "domains": domains,
        "sample_count": len(selected),
        "counts": dict(counts),
        "indices": indices,
    }

    args.subset_output.parent.mkdir(parents=True, exist_ok=True)
    args.indices_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)

    with args.subset_output.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.indices_output.write_text(json.dumps({"indices": indices}, ensure_ascii=False, indent=2), encoding="utf-8")
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
