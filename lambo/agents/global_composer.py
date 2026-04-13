"""GlobalComposer — cross-document evidence synthesis and entity resolution.

Takes per-document evidence sheets from DocRefineAgent and:
1. Builds projection_map (DOC id → real entity name)
2. Cross-document comparison and relation extraction
3. Outputs structured composed view for the Generator
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..backend import QwenLocalClient
from ..common import (
    extract_json_payload,
    json_dumps_pretty,
    read_json,
    write_json,
)


class GlobalComposer:
    def __init__(
        self,
        llm: QwenLocalClient,
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        pdir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts" / "compose"
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
        """Format per-document evidence for the LLM."""
        parts: List[str] = []
        for sheet in doc_sheets:
            doc_id = sheet["doc_id"]
            title = sheet.get("doc_title", "")
            status = sheet.get("scan_result", "no_evidence")
            evidence = sheet.get("evidence", "").strip()

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
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / "composed.json"
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
            max_output_tokens=2500,
            metadata={"module": "global_composer"},
        )

        # Extract projection_map and records from response
        projection_map: Dict[str, str] = {}
        records: List[Any] = []
        structure_description = ""

        if isinstance(payload, dict):
            projection_map = payload.get("projection_map", {})
            records = payload.get("records", [])
            structure_description = str(payload.get("structure_description", ""))

        # Fallback: build projection_map from doc_title if LLM missed it
        if not projection_map:
            for sheet in doc_sheets:
                projection_map[sheet["doc_id"]] = sheet.get("doc_title", sheet["doc_id"])

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
