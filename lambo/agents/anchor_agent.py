"""Anchor Agent — places semantic anchors over a document and builds a document map."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backend import QwenLocalClient
from ..common import (
    compact_text,
    current_query_from_record,
    ensure_dir,
    extract_json_payload,
    instruction_from_record,
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
        max_units_per_doc: int = 220,
    ) -> None:
        self.llm = llm
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts" / "anchor"
        self.system_prompt = (self.prompt_dir / "system.txt").read_text(encoding="utf-8").strip()
        self.user_prompt_template = (self.prompt_dir / "user.txt").read_text(encoding="utf-8").strip()
        self.max_units_per_doc = max_units_per_doc

    def _build_units(self, document_text: str) -> List[Dict[str, Any]]:
        text = (document_text or "").replace("\r\n", "\n")
        raw_blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
        if len(raw_blocks) <= 1:
            raw_blocks = [line.strip() for line in text.splitlines() if line.strip()]

        blocks: List[str] = []
        for block in raw_blocks:
            if len(block) > 2400 and block.count("\n") >= 2:
                blocks.extend(ln.strip() for ln in block.splitlines() if ln.strip())
            else:
                blocks.append(block)

        merged: List[str] = []
        buffer = ""
        for block in blocks:
            if len(block) < 80 and buffer:
                buffer = buffer + "\n" + block
                if len(buffer) >= 160:
                    merged.append(buffer)
                    buffer = ""
            else:
                if buffer:
                    merged.append(buffer)
                buffer = block
        if buffer:
            merged.append(buffer)

        if len(merged) > self.max_units_per_doc:
            target = self.max_units_per_doc
            stride = max(1, len(merged) // target + 1)
            packed: List[str] = []
            group: List[str] = []
            for block in merged:
                group.append(block)
                if len(group) >= stride:
                    packed.append("\n".join(group))
                    group = []
            if group:
                packed.append("\n".join(group))
            merged = packed

        units: List[Dict[str, Any]] = []
        cursor = 0
        for index, block in enumerate(merged, start=1):
            start = text.find(block[:60], cursor) if block else cursor
            if start < 0:
                start = cursor
            end = start + len(block)
            cursor = max(cursor, end)
            units.append(
                {
                    "unit_id": f"U{index:03d}",
                    "order": index,
                    "char_span": [start, end],
                    "text": block,
                    "preview": compact_text(block, limit=220),
                }
            )
        return units

    @staticmethod
    def _serialize_units(units: List[Dict[str, Any]], per_unit_char_limit: int = 480) -> str:
        lines: List[str] = []
        for unit in units:
            body = unit["text"]
            if len(body) > per_unit_char_limit:
                body = body[:per_unit_char_limit].rstrip() + " ..."
            lines.append(f"[{unit['unit_id']}] {body}")
        return "\n".join(lines)

    def _build_anchor_record(
        self,
        *,
        doc_id: str,
        doc_title: str,
        units: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        anchor_index: int,
        anchor_title: str,
        summary: str,
        key_entities: List[str],
    ) -> Dict[str, Any]:
        span = units[start_idx : end_idx + 1]
        text = "\n\n".join(u["text"] for u in span).strip()
        return {
            "anchor_id": f"{doc_id}_A{anchor_index}",
            "doc_id": doc_id,
            "doc_title": doc_title,
            "order": anchor_index,
            "anchor_title": normalize_ws(anchor_title) or f"Anchor {anchor_index}",
            "summary": normalize_ws(summary) or compact_text(text, limit=200),
            "key_entities": [normalize_ws(e) for e in key_entities if normalize_ws(e)][:8],
            "packet_span": f"{span[0]['unit_id']}..{span[-1]['unit_id']}",
            "char_span": [span[0]["char_span"][0], span[-1]["char_span"][1]],
            "unit_ids": [u["unit_id"] for u in span],
            "text": text,
            "preview": compact_text(text, limit=300),
            "prev_anchor_id": "",
            "next_anchor_id": "",
        }

    def _fallback_tile(
        self,
        *,
        doc_id: str,
        doc_title: str,
        units: List[Dict[str, Any]],
        group_size: int = 4,
    ) -> List[Dict[str, Any]]:
        anchors: List[Dict[str, Any]] = []
        idx = 0
        n = len(units)
        while idx < n:
            end = min(idx + group_size - 1, n - 1)
            anchor_index = len(anchors) + 1
            span = units[idx : end + 1]
            title = compact_text(span[0]["text"], limit=50)
            summary = compact_text(" ".join(u["preview"] for u in span), limit=220)
            anchors.append(
                self._build_anchor_record(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    units=units,
                    start_idx=idx,
                    end_idx=end,
                    anchor_index=anchor_index,
                    anchor_title=title,
                    summary=summary,
                    key_entities=[],
                )
            )
            idx = end + 1
        return anchors

    def _anchor_doc_with_llm(
        self,
        *,
        query: str,
        instruction: str,
        doc: Dict[str, Any],
        units: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        user_prompt = self.user_prompt_template.format(
            query=query or "(empty)",
            instruction=instruction or "(none)",
            doc_title=doc["doc_title"],
            unit_count=len(units),
            doc_units_text=self._serialize_units(units),
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=3500,
            metadata={"module": "anchor_agent", "doc_title": doc["doc_title"]},
        )
        payload = extract_json_payload(raw_text)
        if not isinstance(payload, dict) or not isinstance(payload.get("anchors"), list):
            return self._fallback_tile(
                doc_id=doc["doc_id"], doc_title=doc["doc_title"], units=units
            )

        id_to_idx = {u["unit_id"]: i for i, u in enumerate(units)}
        raw_anchors = payload["anchors"]

        parsed: List[Dict[str, Any]] = []
        for item in raw_anchors:
            if not isinstance(item, dict):
                continue
            s_id = normalize_ws(str(item.get("start_unit_id", "")))
            e_id = normalize_ws(str(item.get("end_unit_id", "")))
            if s_id not in id_to_idx or e_id not in id_to_idx:
                continue
            s_idx = id_to_idx[s_id]
            e_idx = id_to_idx[e_id]
            if e_idx < s_idx:
                s_idx, e_idx = e_idx, s_idx
            parsed.append(
                {
                    "start_idx": s_idx,
                    "end_idx": e_idx,
                    "anchor_title": str(item.get("anchor_title", "")),
                    "summary": str(item.get("summary", "")),
                    "key_entities": item.get("key_entities", []) if isinstance(item.get("key_entities", []), list) else [],
                }
            )

        if not parsed:
            return self._fallback_tile(
                doc_id=doc["doc_id"], doc_title=doc["doc_title"], units=units
            )

        parsed.sort(key=lambda p: (p["start_idx"], p["end_idx"]))
        repaired: List[Dict[str, Any]] = []
        cursor = 0
        n = len(units)
        for p in parsed:
            s = max(p["start_idx"], cursor)
            e = max(p["end_idx"], s)
            if s >= n:
                break
            if s > cursor:
                if repaired:
                    repaired[-1]["end_idx"] = s - 1
                else:
                    repaired.append(
                        {
                            "start_idx": cursor,
                            "end_idx": s - 1,
                            "anchor_title": "Preamble",
                            "summary": "",
                            "key_entities": [],
                        }
                    )
            p["start_idx"] = s
            p["end_idx"] = min(e, n - 1)
            repaired.append(p)
            cursor = p["end_idx"] + 1
        if cursor < n:
            if repaired:
                repaired[-1]["end_idx"] = n - 1
            else:
                repaired.append(
                    {
                        "start_idx": 0,
                        "end_idx": n - 1,
                        "anchor_title": "Document",
                        "summary": "",
                        "key_entities": [],
                    }
                )

        anchors: List[Dict[str, Any]] = []
        for i, p in enumerate(repaired, start=1):
            anchors.append(
                self._build_anchor_record(
                    doc_id=doc["doc_id"],
                    doc_title=doc["doc_title"],
                    units=units,
                    start_idx=p["start_idx"],
                    end_idx=p["end_idx"],
                    anchor_index=i,
                    anchor_title=p["anchor_title"],
                    summary=p["summary"],
                    key_entities=p.get("key_entities") or [],
                )
            )

        for i, anchor in enumerate(anchors):
            if i > 0:
                anchor["prev_anchor_id"] = anchors[i - 1]["anchor_id"]
            if i < len(anchors) - 1:
                anchor["next_anchor_id"] = anchors[i + 1]["anchor_id"]
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
        if cache_path.exists() and not force:
            return read_json(cache_path)

        query = current_query_from_record(
            str(record.get("question", "")).strip(),
            str(record.get("instruction", "")).strip(),
        )
        instruction = instruction_from_record(str(record.get("instruction", "")).strip())

        docs = parse_docs_bundle(record.get("docs", ""))
        doc_payloads: List[Dict[str, Any]] = []
        for doc in docs:
            units = self._build_units(doc["content"])
            anchors = self._anchor_doc_with_llm(query=query, instruction=instruction, doc=doc, units=units)
            doc_payloads.append(
                {
                    "doc_id": doc["doc_id"],
                    "doc_title": doc["doc_title"],
                    "unit_count": len(units),
                    "anchor_count": len(anchors),
                    "anchors": anchors,
                    "doc_map": [
                        {
                            "anchor_id": a["anchor_id"],
                            "anchor_title": a["anchor_title"],
                            "summary": a["summary"],
                            "key_entities": a["key_entities"],
                        }
                        for a in anchors
                    ],
                }
            )
            write_json(
                sample_dir / "anchors.partial.json",
                {
                    "record_id": record.get("id"),
                    "progress": {"completed_docs": len(doc_payloads), "total_docs": len(docs)},
                    "docs": doc_payloads,
                },
            )

        payload = {
            "record_id": record.get("id"),
            "sample_id": safe_filename(f"{record.get('type','?')}_level{record.get('level','?')}_{record.get('id','?')}"),
            "query": query,
            "docs": doc_payloads,
        }
        write_json(cache_path, payload)
        return payload
