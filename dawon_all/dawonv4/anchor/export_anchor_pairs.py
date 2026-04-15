from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export anchor text-summary pairs from anchors.json into JSONL and Markdown files."
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Path to anchors.json, a sample directory containing anchors.json, or a run directory containing samples/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing anchor_pairs.jsonl and anchor_pairs.md files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_dirs_from_target(target: Path) -> List[Path]:
    if target.is_file():
        if target.name != "anchors.json":
            raise FileNotFoundError(f"Expected anchors.json file, got: {target}")
        return [target.parent]

    if not target.is_dir():
        raise FileNotFoundError(f"Target not found: {target}")

    if (target / "anchors.json").exists():
        return [target]

    if (target / "samples").is_dir():
        return sorted(
            sample_dir
            for sample_dir in (target / "samples").iterdir()
            if sample_dir.is_dir() and (sample_dir / "anchors.json").exists()
        )

    return sorted(
        sample_dir
        for sample_dir in target.iterdir()
        if sample_dir.is_dir() and (sample_dir / "anchors.json").exists()
    )


def anchor_rows(sample_dir: Path) -> Iterable[Dict[str, Any]]:
    payload = load_json(sample_dir / "anchors.json")
    record_id = payload.get("record_id", "")
    sample_id = payload.get("sample_id", sample_dir.name)
    record_type = payload.get("type", "")
    level = payload.get("level", "")

    for doc in payload.get("docs", []):
        for anchor in doc.get("anchors", []):
            yield {
                "record_id": record_id,
                "sample_id": sample_id,
                "record_type": record_type,
                "level": level,
                "doc_id": doc.get("doc_id", ""),
                "doc_title": doc.get("doc_title", ""),
                "anchor_id": anchor.get("anchor_id", ""),
                "order": anchor.get("order", ""),
                "anchor_type": anchor.get("anchor_type", ""),
                "section_path": anchor.get("section_path", ""),
                "packet_span": anchor.get("packet_span", ""),
                "unit_ids": anchor.get("unit_ids", []),
                "anchor_title": anchor.get("anchor_title", ""),
                "summary": anchor.get("summary", ""),
                "text": anchor.get("text", ""),
            }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_for_rows(sample_dir: Path, rows: List[Dict[str, Any]]) -> str:
    title = f"# Anchor Pairs: {sample_dir.name}"
    blocks: List[str] = [title, ""]
    current_doc_id = None

    for row in rows:
        if row["doc_id"] != current_doc_id:
            current_doc_id = row["doc_id"]
            blocks.append(f"## {row['doc_id']} | {row['doc_title']}")
            blocks.append("")

        blocks.append(f"### {row['anchor_id']}")
        blocks.append(f"- order: {row['order']}")
        blocks.append(f"- anchor_type: {row['anchor_type']}")
        blocks.append(f"- section_path: {row['section_path'] or '(root)'}")
        blocks.append(f"- packet_span: {row['packet_span']}")
        blocks.append(f"- unit_ids: {', '.join(row['unit_ids'])}")
        blocks.append(f"- summary: {row['summary']}")
        blocks.append("")
        blocks.append("```text")
        blocks.append(row["text"])
        blocks.append("```")
        blocks.append("")

    return "\n".join(blocks)


def export_sample(sample_dir: Path, overwrite: bool) -> Dict[str, str]:
    rows = list(anchor_rows(sample_dir))
    jsonl_path = sample_dir / "anchor_pairs.jsonl"
    md_path = sample_dir / "anchor_pairs.md"

    if not overwrite and (jsonl_path.exists() or md_path.exists()):
        return {
            "sample_dir": str(sample_dir),
            "status": "skipped_existing",
            "jsonl_path": str(jsonl_path),
            "md_path": str(md_path),
        }

    write_jsonl(jsonl_path, rows)
    md_path.write_text(markdown_for_rows(sample_dir, rows), encoding="utf-8")
    return {
        "sample_dir": str(sample_dir),
        "status": "written",
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
        "anchor_count": str(len(rows)),
    }


def main() -> None:
    args = parse_args()
    sample_dirs = sample_dirs_from_target(args.target)
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories with anchors.json found under: {args.target}")

    results = [export_sample(sample_dir, overwrite=args.overwrite) for sample_dir in sample_dirs]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
