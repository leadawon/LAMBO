from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backend import QwenLocalClient
from .common import (
    compact_text,
    current_query_from_record,
    ensure_dir,
    extract_json_payload,
    normalize_ws,
    parse_docs_bundle,
    read_json,
    safe_filename,
    write_json,
)


class AnchorAgent:
    def __init__(
        self,
        llm: QwenLocalClient,
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parent / "prompts"
        self.anchor_system_prompt = (self.prompt_dir / "anchor_system.txt").read_text(encoding="utf-8").strip()
        self.anchor_user_prompt = (self.prompt_dir / "anchor_user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _preview(text: str, limit: int = 220) -> str:
        return compact_text(text, limit=limit)

    @staticmethod
    def _unit_type_for_text(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return "blank"
        if re.match(r"^#{1,6}\s+", stripped):
            return "heading"
        if re.match(r"^(part\s+[ivx]+|item\s+\d+\.?)", stripped, flags=re.IGNORECASE):
            return "heading"
        if re.match(r"^(references|bibliography|参考文献)\b", stripped, flags=re.IGNORECASE):
            return "heading"
        if re.match(r"^(第[一二三四五六七八九十百]+|[一二三四五六七八九十]+[、.])", stripped):
            return "heading"
        if stripped.count("|") >= 2:
            return "table"
        if re.match(r"^[-*•]\s+", stripped):
            return "list_item"
        if re.match(r"^\d+[.)、]\s+", stripped):
            return "list_item"
        if re.match(r"^[（(]?[一二三四五六七八九十0-9]+[）)]", stripped):
            return "clause"
        return "paragraph"

    def _build_units(self, document_text: str) -> List[Dict[str, Any]]:
        normalized_text = document_text.replace("\r\n", "\n")
        raw_blocks = [block.strip() for block in re.split(r"\n\s*\n+", normalized_text) if block.strip()]
        if len(raw_blocks) <= 1:
            raw_blocks = [line.strip() for line in normalized_text.splitlines() if line.strip()]

        blocks: List[str] = []
        for block in raw_blocks:
            if len(block) > 2400 and block.count("\n") >= 2:
                blocks.extend(line.strip() for line in block.splitlines() if line.strip())
            else:
                blocks.append(block)

        units: List[Dict[str, Any]] = []
        cursor = 0
        heading_stack: List[str] = []
        for index, block in enumerate(blocks, start=1):
            start = normalized_text.find(block, cursor)
            if start < 0:
                start = cursor
            end = start + len(block)
            cursor = end

            unit_type = self._unit_type_for_text(block)
            stripped = block.strip()
            if unit_type == "heading":
                heading_text = re.sub(r"^#{1,6}\s+", "", stripped).strip() or stripped
                if re.match(r"^#{1,6}\s+", stripped):
                    depth = len(re.match(r"^(#{1,6})\s+", stripped).group(1))  # type: ignore[union-attr]
                elif re.match(r"^(part\s+[ivx]+|item\s+\d+\.?)", stripped, flags=re.IGNORECASE):
                    depth = 2
                elif re.match(r"^第[一二三四五六七八九十百]+", stripped):
                    depth = 2
                elif re.match(r"^[一二三四五六七八九十]+[、.]", stripped):
                    depth = 3
                else:
                    depth = 1
                while len(heading_stack) >= depth:
                    heading_stack.pop()
                heading_stack.append(heading_text)
                section_path = " > ".join(heading_stack)
            else:
                section_path = " > ".join(heading_stack)

            units.append(
                {
                    "unit_id": f"U{index:03d}",
                    "order": index,
                    "unit_type": unit_type,
                    "section_path": section_path,
                    "char_span": [start, end],
                    "text": block,
                    "preview": self._preview(block, limit=240),
                }
            )
        return units

    @staticmethod
    def _serialize_units(units: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for unit in units:
            header = f"{unit['unit_id']} | {unit['unit_type']} | {unit['section_path'] or '(root)'}"
            blocks.append(f"{header}\n{unit['text']}")
        return "\n\n".join(blocks)

    @staticmethod
    def _normalize_anchor_type(anchor_type: str, text: str) -> str:
        normalized = normalize_ws(anchor_type).lower()
        if normalized in {"paragraph_region", "table_region", "clause_region", "attribution_region", "section_header"}:
            return normalized
        lowered = text.casefold()
        if re.search(r"(reference|bibliography|参考文献)", lowered):
            return "attribution_region"
        if "|" in text:
            return "table_region"
        if re.match(r"^[-*•]\s+", text) or re.match(r"^\d+[.)、]\s+", text):
            return "clause_region"
        return "paragraph_region"

    def _build_anchor_record(
        self,
        *,
        doc_id: str,
        doc_title: str,
        units: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        anchor_index: int,
        summary: str,
        anchor_title: str,
        anchor_type: str,
    ) -> Dict[str, Any]:
        span_units = units[start_idx : end_idx + 1]
        start_unit = span_units[0]
        end_unit = span_units[-1]
        text = "\n\n".join(unit["text"] for unit in span_units).strip()
        section_path = normalize_ws(start_unit.get("section_path", "")) or normalize_ws(anchor_title)
        summary = normalize_ws(summary) or f"{anchor_title}: {self._preview(text, limit=180)}"
        return {
            "anchor_id": f"{doc_id}_A{anchor_index}",
            "doc_id": doc_id,
            "doc_title": doc_title,
            "order": anchor_index,
            "section_path": section_path,
            "packet_span": f"{start_unit['unit_id']}..{end_unit['unit_id']}",
            "char_span": [start_unit["char_span"][0], end_unit["char_span"][1]],
            "anchor_type": self._normalize_anchor_type(anchor_type, text),
            "text": text,
            "summary": summary,
            "preview": self._preview(text, limit=300),
            "unit_ids": [unit["unit_id"] for unit in span_units],
            "anchor_title": normalize_ws(anchor_title),
            "prev_anchor_id": "",
            "next_anchor_id": "",
        }

    def _fallback_anchors_from_units(
        self,
        *,
        doc_id: str,
        doc_title: str,
        units: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        anchors: List[Dict[str, Any]] = []
        for index, unit in enumerate(units, start=1):
            anchors.append(
                self._build_anchor_record(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    units=units,
                    start_idx=index - 1,
                    end_idx=index - 1,
                    anchor_index=index,
                    summary=f"{unit['section_path'] or unit['unit_id']}: {self._preview(unit['text'], limit=180)}",
                    anchor_title=unit["section_path"] or unit["unit_id"],
                    anchor_type=unit["unit_type"],
                )
            )
        return anchors

    def _anchor_doc_with_llm(
        self,
        *,
        record: Dict[str, Any],
        doc: Dict[str, Any],
        units: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompt = self.anchor_user_prompt.format(
            record_type=record.get("type"),
            level=record.get("level"),
            question=str(record.get("question", "")).strip() or "(empty)",
            instruction=str(record.get("instruction", "")).strip(),
            current_query=current_query_from_record(
                str(record.get("question", "")).strip(),
                str(record.get("instruction", "")).strip(),
            ),
            doc_title=doc["doc_title"],
            unit_count=len(units),
            doc_units_text=self._serialize_units(units),
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.anchor_system_prompt,
            user_prompt=prompt,
            max_output_tokens=3000,
            metadata={"module": "anchor_agent", "doc_title": doc["doc_title"]},
        )
        payload = extract_json_payload(raw_text)
        if not isinstance(payload, dict) or not isinstance(payload.get("anchors"), list):
            return self._fallback_anchors_from_units(doc_id=doc["doc_id"], doc_title=doc["doc_title"], units=units)

        unit_id_to_index = {unit["unit_id"]: idx for idx, unit in enumerate(units)}
        anchors: List[Dict[str, Any]] = []
        used_spans = set()
        for item in payload["anchors"]:
            if not isinstance(item, dict):
                continue
            start_unit_id = normalize_ws(item.get("start_unit_id", ""))
            end_unit_id = normalize_ws(item.get("end_unit_id", ""))
            if start_unit_id not in unit_id_to_index or end_unit_id not in unit_id_to_index:
                continue
            start_idx = unit_id_to_index[start_unit_id]
            end_idx = unit_id_to_index[end_unit_id]
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            span_key = (start_idx, end_idx)
            if span_key in used_spans:
                continue
            used_spans.add(span_key)
            anchors.append(
                self._build_anchor_record(
                    doc_id=doc["doc_id"],
                    doc_title=doc["doc_title"],
                    units=units,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    anchor_index=len(anchors) + 1,
                    summary=str(item.get("summary", "")),
                    anchor_title=str(item.get("anchor_title", "")),
                    anchor_type=str(item.get("anchor_type", "")),
                )
            )

        if not anchors:
            return self._fallback_anchors_from_units(doc_id=doc["doc_id"], doc_title=doc["doc_title"], units=units)

        anchors.sort(key=lambda anchor: anchor["char_span"][0])
        for index, anchor in enumerate(anchors, start=1):
            anchor["anchor_id"] = f"{doc['doc_id']}_A{index}"
            anchor["order"] = index
            if index > 1:
                anchor["prev_anchor_id"] = anchors[index - 2]["anchor_id"]
            if index < len(anchors):
                anchor["next_anchor_id"] = f"{doc['doc_id']}_A{index + 1}"
        return anchors

    def run(
        self,
        *,
        record: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        ensure_dir(sample_dir)
        cache_path = sample_dir / "anchors.json"
        partial_cache_path = sample_dir / "anchors.partial.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        docs = parse_docs_bundle(record.get("docs", ""))
        doc_payloads: List[Dict[str, Any]] = []

        for doc in docs:
            units = self._build_units(doc["content"])
            anchors = self._anchor_doc_with_llm(record=record, doc=doc, units=units)
            doc_payloads.append(
                {
                    "doc_id": doc["doc_id"],
                    "doc_title": doc["doc_title"],
                    "unit_count": len(units),
                    "anchor_count": len(anchors),
                    "units": [
                        {
                            "unit_id": unit["unit_id"],
                            "unit_type": unit["unit_type"],
                            "section_path": unit["section_path"],
                            "char_span": unit["char_span"],
                            "preview": unit["preview"],
                        }
                        for unit in units
                    ],
                    "anchors": anchors,
                    "anchor_catalog": [
                        {
                            "anchor_id": anchor["anchor_id"],
                            "anchor_type": anchor["anchor_type"],
                            "section_path": anchor["section_path"],
                            "summary": anchor["summary"],
                        }
                        for anchor in anchors
                    ],
                    "search_views": {
                        "anchor_type_counts": {
                            anchor_type: sum(1 for anchor in anchors if anchor["anchor_type"] == anchor_type)
                            for anchor_type in sorted({anchor["anchor_type"] for anchor in anchors})
                        },
                        "anchor_catalog_text": "\n".join(
                            f"- {anchor['anchor_id']} | {anchor['anchor_type']} | {anchor['section_path']} | {anchor['summary']}"
                            for anchor in anchors
                        ),
                    },
                }
            )
            partial_payload = {
                "record_id": record.get("id"),
                "sample_id": safe_filename(f"{record.get('type')}_level{record.get('level')}_{record.get('id')}"),
                "type": record.get("type"),
                "level": record.get("level"),
                "set": record.get("set"),
                "progress": {
                    "completed_docs": len(doc_payloads),
                    "total_docs": len(docs),
                },
                "docs": doc_payloads,
            }
            write_json(partial_cache_path, partial_payload)

        payload = {
            "record_id": record.get("id"),
            "sample_id": safe_filename(f"{record.get('type')}_level{record.get('level')}_{record.get('id')}"),
            "type": record.get("type"),
            "level": record.get("level"),
            "set": record.get("set"),
            "progress": {
                "completed_docs": len(doc_payloads),
                "total_docs": len(docs),
            },
            "docs": doc_payloads,
        }
        write_json(cache_path, payload)
        return payload
