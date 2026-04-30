"""DocRefineAgentV2 — per-document evidence extraction compatible with AnchorAgentV2.

Identical loop to DocRefineAgent (think→search→info) but consumes the TOC-based
doc_payload produced by AnchorAgentV2 instead of the block-based anchor payload.

Inputs per doc: doc_title + toc (list of {number, title, level, char_start, char_end, text}).
The LLM references sections by their dotted number (e.g. "3.2").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backend import GeminiClient, OpenAIClient, QwenLocalClient
from ..common import (
    extract_json_payload,
    extract_tag_content,
    normalize_ws,
    read_json,
    write_json,
)


class DocRefineAgentV2:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
        max_rounds: int = 6,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        pdir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts" / "doc_refine_v2"
        self.system_prompt = (pdir / "system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (pdir / "user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _format_doc_map(toc: List[Dict[str, Any]]) -> str:
        """Render TOC as an indented section list for the LLM."""
        lines: List[str] = []
        for entry in toc:
            number = str(entry.get("number", ""))
            title = str(entry.get("title", ""))
            level = int(entry.get("level", 1))
            indent = "  " * (level - 1)
            lines.append(f"{indent}[{number}] {title}")
        return "\n".join(lines) if lines else "(no sections available)"

    def _call_llm(
        self,
        *,
        question: str,
        instruction: str,
        doc_id: str,
        doc_title: str,
        doc_map_text: str,
        all_docs_list: str,
        accumulated_trace: str,
    ) -> str:
        trace_section = ""
        if accumulated_trace.strip():
            trace_section = f"Previous rounds:\n{accumulated_trace}"
        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            doc_title=doc_title,
            doc_id=doc_id,
            doc_map=doc_map_text,
            all_docs_list=all_docs_list,
            accumulated_trace=trace_section,
        )
        return self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=8192,
            metadata={"module": "doc_refine_v2", "phase": "plan_search", "doc_id": doc_id},
        )

    def run(
        self,
        *,
        question: str,
        instruction: str,
        doc_payload: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
        other_docs: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        doc_id = doc_payload["doc_id"]
        doc_title = doc_payload["doc_title"]
        cache_path = sample_dir / f"{doc_id}_refine.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        toc = doc_payload.get("toc", [])
        doc_map_text = self._format_doc_map(toc)

        # Build lookup: section number → toc entry (for text injection)
        section_by_id: Dict[str, Dict[str, Any]] = {
            str(entry.get("number", "")): entry for entry in toc
        }

        docs_for_list = other_docs or [{"doc_id": doc_id, "doc_title": doc_title}]
        all_docs_list = "\n".join(
            f"- {d['doc_id']}: {d.get('doc_title', '')}" for d in docs_for_list
        )

        opened_sections: List[str] = []
        accumulated_trace = ""

        scan_result = "no_evidence"
        evidence_text = ""

        all_numbers = [str(e.get("number", "")) for e in toc]

        for round_idx in range(1, self.max_rounds + 1):
            raw = self._call_llm(
                question=question,
                instruction=instruction,
                doc_id=doc_id,
                doc_title=doc_title,
                doc_map_text=doc_map_text,
                all_docs_list=all_docs_list,
                accumulated_trace=accumulated_trace,
            )

            think_text = extract_tag_content(raw, "think") or ""
            search_text = extract_tag_content(raw, "search") or ""
            answer_text = extract_tag_content(raw, "answer") or ""
            search_payload = extract_json_payload(search_text)

            accumulated_trace += f"\n--- Round {round_idx} ---\n"
            accumulated_trace += f"<think>{think_text}</think>\n"
            accumulated_trace += f"<search>{search_text}</search>\n"

            action = "stop"
            anchor_id = ""
            if isinstance(search_payload, dict):
                action = str(search_payload.get("action", "stop")).lower()
                anchor_id = normalize_ws(str(search_payload.get("anchor_id", "")))

            if action == "stop":
                if not opened_sections and toc:
                    anchor_id = all_numbers[0]
                    accumulated_trace += (
                        "<note>Stop overridden: must open at least one section before "
                        "judging irrelevance. Forcing open of first section.</note>\n"
                    )
                    action = "open"
                else:
                    if answer_text:
                        accumulated_trace += f"<answer>{answer_text}</answer>\n"
                        evidence_text = answer_text.strip()
                        scan_result = "evidence_found" if evidence_text else "no_evidence"
                    break

            if not anchor_id:
                if not opened_sections and toc:
                    anchor_id = all_numbers[0]
                else:
                    break

            # Validate; fall back to first unopened section
            if anchor_id not in section_by_id or anchor_id in opened_sections:
                fallback_id = ""
                for num in all_numbers:
                    if num not in opened_sections:
                        fallback_id = num
                        break
                if not fallback_id:
                    break
                anchor_id = fallback_id

            opened_sections.append(anchor_id)
            section_text = section_by_id[anchor_id].get("text", "")
            accumulated_trace += f"<info anchor_id=\"{anchor_id}\">\n{section_text}\n</info>\n"

        evidence_sheet = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "scan_result": scan_result,
            "evidence": evidence_text,
            "opened_anchors": opened_sections,
            "rounds_used": len(opened_sections),
            "trace": accumulated_trace,
        }
        write_json(cache_path, evidence_sheet)
        return evidence_sheet
