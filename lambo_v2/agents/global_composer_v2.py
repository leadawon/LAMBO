"""GlobalComposerV2 — human-readable cross-document reasoning trace.

The composer reads the upstream 5W1H evidence records like a careful
reader and emits a tagged plain-text document with five sections:

    <reading_trace>      step-by-step natural-language reasoning
    <evidence_units>     normalised units (U1, U2, …) anchoring records
    <evidence_relations> cross-doc connections in plain language
    <answer_basis>       grounded basis the Generator should use
    <uncertainty>        missing / conflicting / negative notes

The composer never decides the answer's surface format; the Generator
does that from the instruction. The composer's primary product is the
human-readable trace itself — it doubles as the XAI explanation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backend import QwenLocalClient, GeminiClient, OpenAIClient
from ..common import read_json, write_json


COMPOSER_SECTIONS = (
    "reading_trace",
    "evidence_units",
    "evidence_relations",
    "answer_basis",
    "uncertainty",
)


class GlobalComposerV2:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        pdir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts" / "compose_v2"
        self.system_prompt = (pdir / "system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (pdir / "user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _build_doc_title_map(doc_sheets: List[Dict[str, Any]]) -> Dict[str, str]:
        return {
            sheet["doc_id"]: sheet.get("doc_title", sheet["doc_id"])
            for sheet in doc_sheets
        }

    @staticmethod
    def _format_record_card(rec: Dict[str, Any], idx: int) -> str:
        src = rec.get("source") or {}
        rel = rec.get("found_relation")
        lines = [
            f"  Record {idx}:",
            f"    who   : {rec.get('who','')}",
            f"    what  : {rec.get('what','')}",
            f"    when  : {rec.get('when','—')}",
            f"    where : {rec.get('where','')}",
            f"    why   : {rec.get('why','')}",
            f"    how   : {rec.get('how','')}",
            f"    source.section       : {src.get('section','')}",
            f"    source.section_title : {src.get('section_title','')}",
            f"    source.verbatim      : {src.get('verbatim','')!r}",
        ]
        if isinstance(rel, dict):
            lines.append(
                f"    found_relation       : "
                f"{rel.get('subject','')!r} --[{rel.get('predicate','')}]--> "
                f"{rel.get('object','')!r}"
            )
        return "\n".join(lines)

    @staticmethod
    def _prepare_sheets_for_prompt(doc_sheets: List[Dict[str, Any]]) -> str:
        """Render each doc_sheet as a block whose top two lines are the
        rule-based document identity header (doc_id, doc_title) — these
        propagate from the upstream extractor through the composer's
        evidence_units to the generator without paraphrase."""
        parts: List[str] = []
        for sheet in doc_sheets:
            doc_id = sheet["doc_id"]
            title = sheet.get("doc_title", "")
            status = sheet.get("scan_result", "insufficient_evidence")
            records = sheet.get("evidence_records") or []
            verbatim_blob = (sheet.get("evidence", "") or "").strip()

            block = f"=== {doc_id} ===\n"
            block += f"doc_id:    {doc_id}\n"
            block += f"doc_title: {title}\n"
            block += f"status:    {status}\n"

            if isinstance(records, list) and records:
                block += "evidence_records (5W1H):\n"
                for i, rec in enumerate(records, 1):
                    if isinstance(rec, dict):
                        block += GlobalComposerV2._format_record_card(rec, i) + "\n"
            elif verbatim_blob:
                block += f"verbatim_evidence (legacy):\n{verbatim_blob}\n"
            else:
                block += "evidence: (none)\n"
            parts.append(block)
        return "\n".join(parts)

    def run(
        self,
        *,
        question: str,
        instruction: str,
        doc_sheets: List[Dict[str, Any]],
        anchor_docs: Optional[List[Dict[str, Any]]] = None,  # accepted but ignored
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / "composed_v2.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        # The doc_title bundle is intentionally NOT passed as a separate
        # JSON map. doc_id and doc_title appear as rule-based header lines
        # at the top of each evidence sheet so the identity is anchored
        # right next to the evidence itself.
        sheets_text = self._prepare_sheets_for_prompt(doc_sheets)

        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            doc_sheets=sheets_text,
        )

        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=24576,
            metadata={"module": "global_composer_v2"},
        )

        sections = self._parse_sections(raw_text)

        doc_title_lookup = {
            sheet["doc_id"]: sheet.get("doc_title", sheet["doc_id"])
            for sheet in doc_sheets
        }

        result = {
            "reading_trace":      sections.get("reading_trace", ""),
            "evidence_units":     sections.get("evidence_units", ""),
            "evidence_relations": sections.get("evidence_relations", ""),
            "answer_basis":       sections.get("answer_basis", ""),
            "uncertainty":        sections.get("uncertainty", ""),
            "doc_title_lookup":   doc_title_lookup,
            "raw_text":           raw_text,
            "doc_sheet_count":    len(doc_sheets),
            "evidence_found_count": sum(
                1 for s in doc_sheets if s.get("scan_result") == "evidence_found"
            ),
            "scan_result_counts": {
                k: sum(1 for s in doc_sheets if s.get("scan_result") == k)
                for k in (
                    "evidence_found",
                    "negative_evidence_found",
                    "irrelevant_document",
                    "insufficient_evidence",
                )
            },
        }
        write_json(cache_path, result)
        return result

    @staticmethod
    def _parse_sections(text: str) -> Dict[str, str]:
        """Extract <tag>…</tag> blocks for each composer section. Tolerant
        to whitespace and to a missing closing tag (in which case the
        block runs to the next opening tag or end-of-text)."""
        out: Dict[str, str] = {}
        if not isinstance(text, str) or not text.strip():
            return out
        for tag in COMPOSER_SECTIONS:
            pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"
            m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if m:
                out[tag] = m.group(1).strip()
                continue
            # No closing tag — fall back to "from <tag> to next <other_tag> or EOT"
            open_pat = rf"<\s*{tag}\s*>"
            mo = re.search(open_pat, text, flags=re.IGNORECASE)
            if not mo:
                continue
            start = mo.end()
            # find next opening tag of any other section
            next_starts = []
            for other in COMPOSER_SECTIONS:
                if other == tag:
                    continue
                mo2 = re.search(rf"<\s*{other}\s*>", text[start:], flags=re.IGNORECASE)
                if mo2:
                    next_starts.append(start + mo2.start())
            end = min(next_starts) if next_starts else len(text)
            out[tag] = text[start:end].strip()
        return out
