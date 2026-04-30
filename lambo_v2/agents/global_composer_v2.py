"""GlobalComposerV2 — cross-document evidence synthesis without key_entities.

Receives per-document evidence sheets (each with verbatim, information-rich
evidence spans) and synthesizes them into a structured view for the Generator.
No anchor_docs or key_entities — the evidence spans are self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backend import QwenLocalClient, GeminiClient, OpenAIClient
from ..common import read_json, write_json


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
    def _prepare_sheets_for_prompt(doc_sheets: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for sheet in doc_sheets:
            doc_id = sheet["doc_id"]
            title = sheet.get("doc_title", "")
            status = sheet.get("scan_result", "no_evidence")
            evidence = (sheet.get("evidence", "") or "").strip()

            block = f"### {doc_id}: {title}\n"
            block += f"Status: {status}\n"
            if evidence:
                block += f"Evidence:\n{evidence}\n"
            else:
                block += "Evidence: (none)\n"
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

        doc_title_map = self._build_doc_title_map(doc_sheets)
        sheets_text = self._prepare_sheets_for_prompt(doc_sheets)

        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            doc_title_map=json.dumps(doc_title_map, ensure_ascii=False, indent=2),
            doc_sheets_text=sheets_text,
        )

        payload, raw_text = self.llm.generate_json(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=8192,
            metadata={"module": "global_composer_v2"},
        )

        projection_map: Dict[str, str] = {}
        records: List[Any] = []
        structure_description = ""

        if isinstance(payload, dict):
            raw_projection = payload.get("projection_map", {})
            if isinstance(raw_projection, dict):
                projection_map = {str(k): str(v) for k, v in raw_projection.items()}
            raw_records = payload.get("records", [])
            if isinstance(raw_records, list):
                records = raw_records
            elif isinstance(raw_records, dict):
                records = [
                    {"category": str(k), "items": v if isinstance(v, list) else [v]}
                    for k, v in raw_records.items()
                ]
            structure_description = str(payload.get("structure_description", ""))

        if not projection_map:
            projection_map = {
                sheet["doc_id"]: sheet.get("doc_title", sheet["doc_id"])
                for sheet in doc_sheets
            }

        result = {
            "projection_map": projection_map,
            "structure_description": structure_description,
            "records": records,
            "raw_text": raw_text,
            "doc_sheet_count": len(doc_sheets),
            "evidence_found_count": sum(
                1 for s in doc_sheets if s.get("scan_result") == "evidence_found"
            ),
        }
        write_json(cache_path, result)
        return result
