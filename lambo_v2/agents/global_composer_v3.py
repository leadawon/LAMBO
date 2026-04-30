"""GlobalComposerV3 — single-LLM-call composer with explicit reasoning protocol.

The composer makes ONE LLM call that fills five sections in a fixed order:
    1) query_spec       — intent / selector / relator_kind / projector
    2) doc_records      — per-doc verdict + ref_value + facts + outgoing_refs
    3) structure        — fan_in / fan_out / chain / partition / ...
    4) completeness     — expected_hint vs selected_count
    5) filled_skeleton  — ready-to-emit answer for the Generator

Domain-blind: receives only question / instruction / doc titles / evidence.
Post-processing is deterministic Python (no extra LLM call).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backend import GeminiClient, OpenAIClient, QwenLocalClient
from ..common import read_json, write_json


class GlobalComposerV3:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        pdir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts" / "compose_v3"
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
        cache_path = sample_dir / "composed_v3.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        doc_title_map = self._build_doc_title_map(doc_sheets)
        sheets_text = self._prepare_sheets_for_prompt(doc_sheets)

        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            doc_title_map_json=json.dumps(doc_title_map, ensure_ascii=False, indent=2),
            doc_sheets_text=sheets_text,
        )

        payload, raw_text = self.llm.generate_json(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=8192,
            metadata={"module": "global_composer_v3"},
        )

        result = self._normalize_and_verify(payload, doc_sheets, raw_text)
        write_json(cache_path, result)
        return result

    @staticmethod
    def _coerce_dict(obj: Any) -> Dict[str, Any]:
        return obj if isinstance(obj, dict) else {}

    @staticmethod
    def _coerce_list(obj: Any) -> List[Any]:
        return obj if isinstance(obj, list) else []

    def _normalize_and_verify(
        self,
        payload: Any,
        doc_sheets: List[Dict[str, Any]],
        raw_text: str,
    ) -> Dict[str, Any]:
        payload = self._coerce_dict(payload)

        query_spec = self._coerce_dict(payload.get("query_spec"))
        projector = self._coerce_dict(query_spec.get("projector"))
        ref_unit = str(projector.get("ref_unit", "doc_id")).strip().lower() or "doc_id"

        records_raw = self._coerce_list(payload.get("doc_records"))
        records: List[Dict[str, Any]] = []
        for rec in records_raw:
            rec = self._coerce_dict(rec)
            doc_id = str(rec.get("doc_id", "")).strip()
            if not doc_id:
                continue
            verdict = str(rec.get("verdict", "no_evidence")).strip().lower()
            if verdict not in {"match", "distractor", "no_evidence"}:
                verdict = "no_evidence"
            records.append({
                "doc_id": doc_id,
                "verdict": verdict,
                "verdict_reason": str(rec.get("verdict_reason", "")),
                "ref_value": str(rec.get("ref_value", doc_id)),
                "facts": str(rec.get("facts", "")),
                "outgoing_refs": self._coerce_list(rec.get("outgoing_refs")),
            })

        # Ensure every input doc has a record (LLM may drop one)
        existing_ids = {r["doc_id"] for r in records}
        order_index = {sheet["doc_id"]: i for i, sheet in enumerate(doc_sheets)}
        for sheet in doc_sheets:
            if sheet["doc_id"] not in existing_ids:
                records.append({
                    "doc_id": sheet["doc_id"],
                    "verdict": "no_evidence",
                    "verdict_reason": "(auto-filled — LLM omitted this doc)",
                    "ref_value": sheet["doc_id"],
                    "facts": "",
                    "outgoing_refs": [],
                })
        records.sort(key=lambda r: order_index.get(r["doc_id"], len(doc_sheets)))

        structure = self._coerce_dict(payload.get("structure"))
        completeness = self._coerce_dict(payload.get("completeness"))

        # Verify selected_count
        match_records = [r for r in records if r["verdict"] == "match"]
        actual_count = len(match_records)
        warnings: List[str] = []
        try:
            stated = int(completeness.get("selected_count", actual_count))
        except (TypeError, ValueError):
            stated = actual_count
        if stated != actual_count:
            warnings.append(
                f"selected_count mismatch: stated={stated}, actual={actual_count}"
            )
        completeness["selected_count"] = actual_count

        filled_skeleton = payload.get("filled_skeleton", None)

        result: Dict[str, Any] = {
            "query_spec": query_spec,
            "doc_records": records,
            "structure": structure,
            "completeness": completeness,
            "filled_skeleton": filled_skeleton,
            "warnings": warnings,
            "raw_text": raw_text,
            "doc_sheet_count": len(doc_sheets),
            "evidence_found_count": sum(
                1 for s in doc_sheets if s.get("scan_result") == "evidence_found"
            ),
        }

        # Backward-compat fields so the existing Generator keeps working
        # without modification. These mirror the v2 composer's output.
        result["projection_map"] = {
            r["doc_id"]: r["ref_value"] for r in records
        }
        result["records"] = match_records
        result["structure_description"] = str(structure.get("summary", ""))
        result["ref_unit"] = ref_unit

        return result
