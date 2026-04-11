from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backend import QwenLocalClient
from .common import (
    answer_topology_for_record,
    compact_text,
    normalize_mapping_values,
    normalize_ws,
    read_json,
    write_json,
)


class AnswerWriter:
    def __init__(self, llm: QwenLocalClient) -> None:
        self.llm = llm

    @staticmethod
    def _stringify_items(doc_results: List[Dict[str, Any]]) -> str:
        rows: List[Dict[str, Any]] = []
        for result in doc_results:
            rows.append(
                {
                    "doc_id": result["doc_id"],
                    "doc_title": result["doc_title"],
                    "items": result.get("items", []),
                }
            )
        return json.dumps(rows, ensure_ascii=False, indent=2)

    @staticmethod
    def _fallback_answer(record: Dict[str, Any], doc_results: List[Dict[str, Any]]) -> Any:
        topology = answer_topology_for_record(record)
        record_type = str(record.get("type", "")).strip().lower()
        level = int(record.get("level", 0) or 0)
        flattened: List[Dict[str, Any]] = []
        for result in doc_results:
            flattened.extend(result.get("items", []))

        if topology == "str":
            values = []
            ranked_items = sorted(flattened, key=lambda entry: float(entry.get("confidence", 0.0)), reverse=True)
            for item in sorted(flattened, key=lambda entry: float(entry.get("confidence", 0.0)), reverse=True):
                normalized = normalize_ws(item.get("normalized_value", "")) or normalize_ws(item.get("value", ""))
                if normalized and normalized not in values:
                    values.append(normalized)
            if not values:
                return ""
            if (record_type, level) in {("financial", 1), ("legal", 1)}:
                return values[0]
            if (record_type, level) in {("financial", 2), ("legal", 2)}:
                return "，".join(values)
            if level == 4 and ranked_items:
                return values[0]
            if len(values) == 1:
                return values[0]
            return "，".join(values)

        if topology == "dict":
            grouped: Dict[str, List[str]] = {}
            for item in flattened:
                key = normalize_ws(item.get("answer_key", ""))
                if not key:
                    continue
                value = item.get("normalized_value", item.get("value", ""))
                normalized_values = normalize_mapping_values(value)
                if not normalized_values and isinstance(value, list):
                    normalized_values = normalize_mapping_values(value)
                grouped.setdefault(key, [])
                for entry in normalized_values:
                    if entry not in grouped[key]:
                        grouped[key].append(entry)
            return grouped

        values = []
        for item in sorted(flattened, key=lambda entry: float(entry.get("confidence", 0.0)), reverse=True):
            value = item.get("normalized_value", item.get("value", ""))
            if isinstance(value, dict):
                for entry in value.values():
                    if isinstance(entry, str) and entry not in values:
                        values.append(entry)
            elif isinstance(value, list):
                for entry in value:
                    if str(entry) not in values:
                        values.append(str(entry))
            else:
                normalized = normalize_ws(value)
                if normalized and normalized not in values:
                    values.append(normalized)
        return values

    @staticmethod
    def _coerce_output(topology: str, payload: Any) -> Any:
        if isinstance(payload, dict) and "final_answer" in payload:
            payload = payload["final_answer"]
        if topology == "str":
            if isinstance(payload, list):
                return "，".join(normalize_ws(item) for item in payload if normalize_ws(item))
            if isinstance(payload, dict):
                return normalize_ws(json.dumps(payload, ensure_ascii=False))
            return normalize_ws(payload)
        if topology == "dict":
            if isinstance(payload, dict):
                return payload
            return {}
        if topology == "list":
            if isinstance(payload, list):
                return payload
            if isinstance(payload, str):
                return [normalize_ws(payload)] if normalize_ws(payload) else []
            return []
        return payload

    def run(
        self,
        *,
        record: Dict[str, Any],
        doc_results: List[Dict[str, Any]],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / "answer_writer.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        topology = answer_topology_for_record(record)
        extracted_items_json = self._stringify_items(doc_results)
        system_prompt = (
            "You are the final LAMBO answer writer. Only use extracted evidence items. "
            "Do not hallucinate and do not use outside knowledge. "
            "Return strict JSON only in the form {\"final_answer\": ...}."
        )
        user_prompt = (
            f"Record type: {record.get('type')}\n"
            f"Level: {record.get('level')}\n"
            f"Answer topology: {topology}\n"
            f"Question: {str(record.get('question', '')).strip() or '(empty)'}\n"
            f"Instruction:\n{str(record.get('instruction', '')).strip()}\n\n"
            f"Extracted doc evidence items:\n{extracted_items_json}\n\n"
            "Compose the final answer only from the extracted evidence items.\n"
            "For string answers, preserve the benchmark style as much as possible.\n"
            "For dict/list answers, output valid JSON values."
        )
        payload, raw_text = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=1200,
            metadata={"module": "answer_writer", "record_id": record.get("id")},
        )
        final_answer = self._coerce_output(topology, payload)
        if final_answer == "" or final_answer == {} or final_answer == []:
            final_answer = self._fallback_answer(record, doc_results)
        result = {
            "topology": topology,
            "final_answer": final_answer,
            "raw_text": raw_text,
            "evidence_item_count": sum(len(result.get("items", [])) for result in doc_results),
        }
        write_json(cache_path, result)
        return result
