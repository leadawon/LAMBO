"""Relation Refiner — fuses per-document evidence into one structured view.

Takes the union of atomic facts produced by the per-document Search Agents and
asks the LLM to filter them, pick the structural representation that best fits
the query, and emit a strict JSON "relations" object (table / mapping / graph /
list / scalar).  The Answer Writer then composes the final answer from this
single structured view.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backend import QwenLocalClient
from .common import (
    extract_json_payload,
    extract_tag_content,
    normalize_ws,
    read_json,
    write_json,
)


def _format_facts(doc_results: List[Dict[str, Any]], max_chars_per_fact: int = 400) -> str:
    lines: List[str] = []
    idx = 0
    for doc in doc_results:
        doc_id = doc.get("doc_id", "")
        doc_title = doc.get("doc_title", "")
        lines.append(f"### {doc_id} — {doc_title}")
        for item in doc.get("items", []):
            idx += 1
            fact = str(item.get("fact", ""))[:max_chars_per_fact]
            ents = ", ".join(item.get("entities") or [])
            span = str(item.get("evidence_span", ""))[:200]
            src = item.get("source_anchor_id", "")
            lines.append(
                f"{idx}. [{doc_id}:{src}] fact: {fact}"
                + (f"\n    entities: {ents}" if ents else "")
                + (f"\n    span: {span}" if span else "")
            )
        if not doc.get("items"):
            lines.append("  (no facts)")
        lines.append("")
    return "\n".join(lines).strip() or "(no facts at all)"


class RelationRefiner:
    def __init__(
        self,
        llm: QwenLocalClient,
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parent / "prompts"
        self.system_prompt = (self.prompt_dir / "refine_system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (self.prompt_dir / "refine_user.txt").read_text(encoding="utf-8").strip()

    def run(
        self,
        *,
        query: str,
        instruction: str,
        doc_results: List[Dict[str, Any]],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / "relations.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        all_facts_text = _format_facts(doc_results)
        user_prompt = self.user_template.format(
            query=query or "(empty)",
            instruction=instruction or "(none)",
            all_facts=all_facts_text,
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=2200,
            metadata={"module": "relation_refiner"},
        )
        think_text = normalize_ws(extract_tag_content(raw_text, "think") or "")
        rel_text = extract_tag_content(raw_text, "relations") or ""
        payload = extract_json_payload(rel_text)
        if not isinstance(payload, dict):
            payload = {"structure": "scalar", "schema": {}, "records": []}
        payload.setdefault("structure", "scalar")
        payload.setdefault("schema", {})
        payload.setdefault("records", [])
        result = {
            "think": think_text,
            "relations": payload,
            "raw_text": raw_text,
        }
        write_json(cache_path, result)
        return result
