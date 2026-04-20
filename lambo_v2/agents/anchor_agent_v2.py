"""Anchor Agent v2 — structure-preserving anchor placement over raw documents.

This version differs from the original v2 anchor agent in one crucial way:
the LLM is no longer tiling over heuristic fixed-size units. Instead, we first
extract larger *structural blocks* that try to preserve document-native
boundaries such as headings, tables, lists, references, and intact paragraphs.
The LLM then chooses anchor boundaries over those contiguous block ranges.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..backend import GeminiClient, OpenAIClient, QwenLocalClient
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
    split_long_paragraph,
    write_json,
)


DEFAULT_SYSTEM_PROMPT = """
You are the Anchor Agent v2.

Your job is to place anchors over ONE document so a downstream agent can search
it without reading everything, while still preserving document structure and
evidence locality.

Important:
- The document below is already shown as numbered structural blocks `B001, B002, ...`.
- These blocks are NOT arbitrary token windows. They were built to preserve raw
  structure such as headings, table row groups, list items, reference entries,
  and intact paragraphs.
- You must decide the anchor boundaries yourself by grouping contiguous block
  ranges into coherent regions.

What an anchor should preserve:
- a section or subsection
- a table together with its header / row group
- a numbered clause or list item sequence
- a bibliographic entry or short run of closely related entries
- a single case-fact block / verdict block / reasoning block
- one locally coherent paragraph group stating one idea

Rules:
- Tile the full document from B001 to the last block with no gaps and no overlaps.
- Never cut a table in half. If several adjacent table blocks share one header or
  belong to the same statement, keep them inside one anchor.
- Never split one bibliographic entry or one numbered clause across anchors.
- Keep numerical values, named entities, years, verdict labels, metrics, and
  explicit relations inside the same anchor whenever possible.
- Target roughly 5–20 anchors per document, but coherence is more important than
  hitting an exact count.
- `summary` must be faithful and evidence-centric. Preserve concrete names,
  numbers, years, and relation clues verbatim when helpful.
- `region_type` is a short descriptive tag such as `heading`, `table`,
  `financial_statement`, `case_facts`, `verdict`, `references`, `abstract`,
  `method`, `results`, `narrative`, `boilerplate`.
- `heading_path` should use document wording if visible. If not visible, provide
  a short path based on the region itself.

Output strict JSON only:
{
  "anchors": [
    {
      "anchor_title": "short label",
      "start_block_id": "B001",
      "end_block_id": "B004",
      "summary": "2-6 sentences, faithful and evidence-centric",
      "region_type": "short tag",
      "heading_path": ["parent heading", "child heading"],
      "key_entities": ["verbatim entity", "verbatim number"]
    }
  ]
}
""".strip()


DEFAULT_USER_PROMPT = """
Question (orientation only — do not tailor anchors narrowly to it):
{query}

Instruction (orientation only):
{instruction}

Document title:
{doc_title}

Number of structural blocks: {block_count}

Document blocks (each block is raw document text with its id, type, and line span):
{doc_blocks_text}

Return strict JSON only. The anchors must cover every block from B001 to the last
block exactly once with no gaps and no overlaps.
""".strip()


class AnchorAgentV2:
    def __init__(self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
        max_blocks_per_doc: int = 320,
        max_block_chars: int = 2200,
        fallback_target_chars: int = 3200,
    ) -> None:
        self.llm = llm
        self.prompt_dir = prompt_dir
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = DEFAULT_USER_PROMPT
        if prompt_dir is not None:
            system_path = prompt_dir / "system.txt"
            user_path = prompt_dir / "user.txt"
            if system_path.exists():
                self.system_prompt = system_path.read_text(encoding="utf-8").strip()
            if user_path.exists():
                self.user_prompt_template = user_path.read_text(encoding="utf-8").strip()
        self.max_blocks_per_doc = max_blocks_per_doc
        self.max_block_chars = max_block_chars
        self.fallback_target_chars = fallback_target_chars

    @staticmethod
    def _line_offsets(text: str) -> Tuple[List[str], List[int]]:
        lines = text.split("\n")
        starts: List[int] = []
        cursor = 0
        for line in lines:
            starts.append(cursor)
            cursor += len(line) + 1
        return lines, starts

    @staticmethod
    def _is_heading(line: str) -> bool:
        s = line.strip()
        if not s or len(s) > 120:
            return False
        if re.fullmatch(r"(?:第[一二三四五六七八九十百零]+[章节部分篇]|[一二三四五六七八九十]+[、.．]|[(（][一二三四五六七八九十0-9]+[)）]|[0-9]+[.．、])\s*.*", s):
            return True
        if re.fullmatch(r"[A-Z][A-Z0-9 .:/_-]{2,}", s):
            return True
        keywords = ("摘要", "引言", "前言", "结论", "附录", "参考文献", "审理查明", "本院认为", "裁定如下")
        return any(s.startswith(keyword) for keyword in keywords)

    @staticmethod
    def _is_reference_entry(line: str) -> bool:
        s = line.strip()
        return bool(
            s
            and (
                re.match(r"^\[[0-9]{1,3}\]\s+", s)
                or re.match(r"^[0-9]{1,3}\.\s+", s)
                or re.match(r"^[A-Z][A-Za-z'`-]+,\s*[A-Z]\.", s)
            )
        )

    @staticmethod
    def _is_list_item(line: str) -> bool:
        s = line.strip()
        return bool(
            s
            and re.match(
                r"^(?:[-*•▪◦]|\(?[0-9]{1,3}\)|[0-9]{1,3}[.)]|[a-zA-Z][.)]|[一二三四五六七八九十]+[、.．])\s+",
                s,
            )
        )

    @staticmethod
    def _is_table_like(line: str) -> bool:
        s = line.rstrip()
        if not s:
            return False
        if "|" in s or "\t" in s:
            return True
        if re.search(r"\b(?:项目|期末余额|期初余额|本期|上期|金额|比例|单位)\b", s) and re.search(r"\d", s):
            return True
        separators = len(re.findall(r"\s{2,}", s))
        numeric_cells = len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?%?", s))
        return separators >= 2 and numeric_cells >= 1

    def _line_kind(self, line: str) -> str:
        if self._is_heading(line):
            return "heading"
        if self._is_reference_entry(line):
            return "reference_entry"
        if self._is_table_like(line):
            return "table"
        if self._is_list_item(line):
            return "list_item"
        return "paragraph"

    def _make_block(
        self,
        *,
        text: str,
        kind: str,
        line_start: int,
        line_end: int,
        char_start: int,
        char_end: int,
    ) -> Dict[str, Any]:
        return {
            "block_id": "",
            "block_type": kind,
            "line_span": [line_start + 1, line_end + 1],
            "char_span": [char_start, char_end],
            "text": text,
            "preview": compact_text(text, limit=260),
        }

    def _split_overlong_block(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = block["text"]
        if len(text) <= self.max_block_chars or block["block_type"] not in {"paragraph", "list_item"}:
            return [block]

        pieces = split_long_paragraph(text, target_chars=self.max_block_chars)
        if len(pieces) <= 1:
            return [block]

        result: List[Dict[str, Any]] = []
        local_cursor = 0
        base_char = block["char_span"][0]
        line_start = block["line_span"][0] - 1
        line_end = block["line_span"][1] - 1
        for piece in pieces:
            idx = text.find(piece, local_cursor)
            if idx < 0:
                idx = local_cursor
            char_start = base_char + idx
            char_end = char_start + len(piece)
            local_cursor = idx + len(piece)
            result.append(
                self._make_block(
                    text=piece,
                    kind=block["block_type"],
                    line_start=line_start,
                    line_end=line_end,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        return result

    def _coarsen_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(blocks) <= self.max_blocks_per_doc:
            return blocks

        merged: List[Dict[str, Any]] = []
        pending: Optional[Dict[str, Any]] = None
        for block in blocks:
            if pending is None:
                pending = dict(block)
                continue

            compatible = pending["block_type"] in {"paragraph", "list_item"} and block["block_type"] == pending["block_type"]
            short_enough = len(pending["text"]) + len(block["text"]) <= self.fallback_target_chars
            if compatible and short_enough:
                pending["text"] = pending["text"].rstrip() + "\n" + block["text"].lstrip()
                pending["char_span"][1] = block["char_span"][1]
                pending["line_span"][1] = block["line_span"][1]
                pending["preview"] = compact_text(pending["text"], limit=260)
            else:
                merged.append(pending)
                pending = dict(block)
        if pending is not None:
            merged.append(pending)
        return merged

    def _build_structural_blocks(self, document_text: str) -> List[Dict[str, Any]]:
        text = (document_text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines, line_starts = self._line_offsets(text)
        blocks: List[Dict[str, Any]] = []
        index = 0

        while index < len(lines):
            raw_line = lines[index]
            if not raw_line.strip():
                index += 1
                continue

            kind = self._line_kind(raw_line)
            start = index
            bucket = [raw_line]
            index += 1

            if kind == "heading":
                pass
            elif kind == "table":
                while index < len(lines):
                    line = lines[index]
                    if not line.strip() or self._line_kind(line) != "table":
                        break
                    bucket.append(line)
                    index += 1
            elif kind == "reference_entry":
                while index < len(lines):
                    line = lines[index]
                    if not line.strip():
                        break
                    next_kind = self._line_kind(line)
                    if next_kind in {"heading", "table"} or self._is_reference_entry(line):
                        break
                    bucket.append(line)
                    index += 1
            elif kind == "list_item":
                while index < len(lines):
                    line = lines[index]
                    if not line.strip():
                        break
                    next_kind = self._line_kind(line)
                    if next_kind in {"heading", "table", "reference_entry"} or self._is_list_item(line):
                        break
                    bucket.append(line)
                    index += 1
            else:
                while index < len(lines):
                    line = lines[index]
                    if not line.strip():
                        break
                    next_kind = self._line_kind(line)
                    if next_kind in {"heading", "table", "reference_entry", "list_item"}:
                        break
                    bucket.append(line)
                    index += 1

            end = start + len(bucket) - 1
            char_start = line_starts[start]
            char_end = line_starts[end] + len(lines[end])
            block = self._make_block(
                text="\n".join(bucket).strip(),
                kind=kind,
                line_start=start,
                line_end=end,
                char_start=char_start,
                char_end=char_end,
            )
            blocks.extend(self._split_overlong_block(block))

        blocks = self._coarsen_blocks(blocks)
        for block_index, block in enumerate(blocks, start=1):
            block["block_id"] = f"B{block_index:03d}"
        return blocks

    @staticmethod
    def _serialize_blocks(blocks: List[Dict[str, Any]]) -> str:
        chunks: List[str] = []
        for block in blocks:
            line_span = f"L{block['line_span'][0]}-L{block['line_span'][1]}"
            chunks.append(
                f"[{block['block_id']}] type={block['block_type']} lines={line_span}\n{block['text']}"
            )
        return "\n\n".join(chunks)

    def _build_anchor_record(
        self,
        *,
        doc_id: str,
        doc_title: str,
        blocks: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        anchor_index: int,
        anchor_title: str,
        summary: str,
        key_entities: List[str],
        region_type: str,
        heading_path: List[str],
    ) -> Dict[str, Any]:
        span = blocks[start_idx : end_idx + 1]
        text = "\n\n".join(block["text"] for block in span).strip()
        clean_heading = [normalize_ws(item) for item in heading_path if normalize_ws(item)][:4]
        return {
            "anchor_id": f"{doc_id}_A{anchor_index}",
            "doc_id": doc_id,
            "doc_title": doc_title,
            "order": anchor_index,
            "anchor_title": normalize_ws(anchor_title) or f"Anchor {anchor_index}",
            "summary": normalize_ws(summary) or compact_text(text, limit=200),
            "key_entities": [normalize_ws(entity) for entity in key_entities if normalize_ws(entity)][:8],
            "packet_span": f"{span[0]['block_id']}..{span[-1]['block_id']}",
            "char_span": [span[0]["char_span"][0], span[-1]["char_span"][1]],
            "line_span": [span[0]["line_span"][0], span[-1]["line_span"][1]],
            "block_ids": [block["block_id"] for block in span],
            "text": text,
            "preview": compact_text(text, limit=300),
            "region_type": normalize_ws(region_type) or span[0]["block_type"],
            "heading_path": clean_heading or [normalize_ws(anchor_title) or span[0]["block_type"]],
            "navigation_summary": normalize_ws(summary) or compact_text(text, limit=200),
            "prev_anchor_id": "",
            "next_anchor_id": "",
        }

    def _fallback_cover(self, *, doc_id: str, doc_title: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        anchors: List[Dict[str, Any]] = []
        start = 0
        current_chars = 0
        for idx, block in enumerate(blocks):
            block_chars = len(block["text"])
            should_break = (
                idx > start
                and current_chars >= self.fallback_target_chars
                and block["block_type"] in {"heading", "table", "reference_entry"}
            )
            if should_break:
                anchor_index = len(anchors) + 1
                anchors.append(
                    self._build_anchor_record(
                        doc_id=doc_id,
                        doc_title=doc_title,
                        blocks=blocks,
                        start_idx=start,
                        end_idx=idx - 1,
                        anchor_index=anchor_index,
                        anchor_title=compact_text(blocks[start]["text"], limit=50),
                        summary=compact_text(
                            " ".join(block_item["preview"] for block_item in blocks[start:idx]),
                            limit=240,
                        ),
                        key_entities=[],
                        region_type=blocks[start]["block_type"],
                        heading_path=[blocks[start]["block_type"]],
                    )
                )
                start = idx
                current_chars = 0
            current_chars += block_chars

        if start < len(blocks):
            anchor_index = len(anchors) + 1
            anchors.append(
                self._build_anchor_record(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    blocks=blocks,
                    start_idx=start,
                    end_idx=len(blocks) - 1,
                    anchor_index=anchor_index,
                    anchor_title=compact_text(blocks[start]["text"], limit=50),
                    summary=compact_text(
                        " ".join(block["preview"] for block in blocks[start:]),
                        limit=240,
                    ),
                    key_entities=[],
                    region_type=blocks[start]["block_type"],
                    heading_path=[blocks[start]["block_type"]],
                )
            )

        for i, anchor in enumerate(anchors):
            if i > 0:
                anchor["prev_anchor_id"] = anchors[i - 1]["anchor_id"]
            if i < len(anchors) - 1:
                anchor["next_anchor_id"] = anchors[i + 1]["anchor_id"]
        return anchors

    def _repair_tiling(self, parsed: List[Dict[str, Any]], blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parsed.sort(key=lambda item: (item["start_idx"], item["end_idx"]))
        repaired: List[Dict[str, Any]] = []
        cursor = 0
        total = len(blocks)

        for item in parsed:
            start_idx = max(item["start_idx"], cursor)
            end_idx = max(item["end_idx"], start_idx)
            if start_idx >= total:
                break

            if start_idx > cursor:
                repaired.append(
                    {
                        "start_idx": cursor,
                        "end_idx": start_idx - 1,
                        "anchor_title": "Bridge",
                        "summary": "",
                        "key_entities": [],
                        "region_type": "",
                        "heading_path": [],
                    }
                )

            repaired.append(
                {
                    **item,
                    "start_idx": start_idx,
                    "end_idx": min(end_idx, total - 1),
                }
            )
            cursor = min(end_idx, total - 1) + 1

        if cursor < total:
            repaired.append(
                {
                    "start_idx": cursor,
                    "end_idx": total - 1,
                    "anchor_title": "Remainder",
                    "summary": "",
                    "key_entities": [],
                    "region_type": "",
                    "heading_path": [],
                }
            )

        return repaired

    def _anchor_doc_with_llm(
        self,
        *,
        query: str,
        instruction: str,
        doc: Dict[str, Any],
        blocks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str]:
        user_prompt = self.user_prompt_template.format(
            query=query or "(empty)",
            instruction=instruction or "(none)",
            doc_title=doc["doc_title"],
            block_count=len(blocks),
            doc_blocks_text=self._serialize_blocks(blocks),
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=4000,
            metadata={"module": "anchor_agent_v2", "doc_title": doc["doc_title"]},
        )

        payload = extract_json_payload(raw_text)
        if not isinstance(payload, dict) or not isinstance(payload.get("anchors"), list):
            return self._fallback_cover(doc_id=doc["doc_id"], doc_title=doc["doc_title"], blocks=blocks), raw_text

        id_to_idx = {block["block_id"]: idx for idx, block in enumerate(blocks)}
        parsed: List[Dict[str, Any]] = []
        for item in payload["anchors"]:
            if not isinstance(item, dict):
                continue
            start_id = normalize_ws(str(item.get("start_block_id", "")))
            end_id = normalize_ws(str(item.get("end_block_id", "")))
            if start_id not in id_to_idx or end_id not in id_to_idx:
                continue
            start_idx = id_to_idx[start_id]
            end_idx = id_to_idx[end_id]
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            heading_path = item.get("heading_path", [])
            if isinstance(heading_path, str):
                heading_path = [heading_path]
            elif not isinstance(heading_path, list):
                heading_path = []
            key_entities = item.get("key_entities", [])
            if not isinstance(key_entities, list):
                key_entities = []
            parsed.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "anchor_title": str(item.get("anchor_title", "")),
                    "summary": str(item.get("summary", "")),
                    "key_entities": [str(entity) for entity in key_entities],
                    "region_type": str(item.get("region_type", "")),
                    "heading_path": [str(path) for path in heading_path],
                }
            )

        if not parsed:
            return self._fallback_cover(doc_id=doc["doc_id"], doc_title=doc["doc_title"], blocks=blocks), raw_text

        repaired = self._repair_tiling(parsed, blocks)
        anchors: List[Dict[str, Any]] = []
        for anchor_index, item in enumerate(repaired, start=1):
            anchors.append(
                self._build_anchor_record(
                    doc_id=doc["doc_id"],
                    doc_title=doc["doc_title"],
                    blocks=blocks,
                    start_idx=item["start_idx"],
                    end_idx=item["end_idx"],
                    anchor_index=anchor_index,
                    anchor_title=item["anchor_title"],
                    summary=item["summary"],
                    key_entities=item.get("key_entities") or [],
                    region_type=item.get("region_type", ""),
                    heading_path=item.get("heading_path") or [],
                )
            )

        for i, anchor in enumerate(anchors):
            if i > 0:
                anchor["prev_anchor_id"] = anchors[i - 1]["anchor_id"]
            if i < len(anchors) - 1:
                anchor["next_anchor_id"] = anchors[i + 1]["anchor_id"]
        return anchors, raw_text

    def run(
        self,
        *,
        record: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        ensure_dir(sample_dir)
        cache_path = sample_dir / "anchors_v2.json"
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
            blocks = self._build_structural_blocks(doc["content"])
            anchors, raw_response = self._anchor_doc_with_llm(
                query=query,
                instruction=instruction,
                doc=doc,
                blocks=blocks,
            )
            doc_payloads.append(
                {
                    "doc_id": doc["doc_id"],
                    "doc_title": doc["doc_title"],
                    "unit_count": len(blocks),
                    "block_count": len(blocks),
                    "anchor_count": len(anchors),
                    "block_map": [
                        {
                            "block_id": block["block_id"],
                            "block_type": block["block_type"],
                            "line_span": block["line_span"],
                            "preview": block["preview"],
                        }
                        for block in blocks
                    ],
                    "anchors": anchors,
                    "doc_map": [
                        {
                            "anchor_id": anchor["anchor_id"],
                            "anchor_title": anchor["anchor_title"],
                            "summary": anchor["summary"],
                            "key_entities": anchor["key_entities"],
                        }
                        for anchor in anchors
                    ],
                    "anchor_generation_raw": raw_response,
                }
            )
            write_json(
                sample_dir / "anchors_v2.partial.json",
                {
                    "record_id": record.get("id"),
                    "progress": {"completed_docs": len(doc_payloads), "total_docs": len(docs)},
                    "docs": doc_payloads,
                },
            )

        payload = {
            "record_id": record.get("id"),
            "sample_id": safe_filename(
                f"{record.get('type', '?')}_level{record.get('level', '?')}_{record.get('id', '?')}"
            ),
            "query": query,
            "segmentation_mode": "structure_preserving_blocks",
            "docs": doc_payloads,
        }
        write_json(cache_path, payload)
        return payload
