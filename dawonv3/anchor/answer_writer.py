from __future__ import annotations

import json
import re
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
    def _flatten_items(doc_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flattened: List[Dict[str, Any]] = []
        for result in doc_results:
            for item in result.get("items", []):
                item_with_doc = dict(item)
                item_with_doc.setdefault("doc_title", result.get("doc_title", ""))
                flattened.append(item_with_doc)
        return flattened

    @staticmethod
    def _is_doc_placeholder(value: Any) -> bool:
        text = normalize_ws(value)
        return bool(re.fullmatch(r"DOC\d+", text, flags=re.IGNORECASE))

    @staticmethod
    def _doc_title_answer(doc_title: str) -> str:
        title = normalize_ws(doc_title)
        cleaned = re.sub(r"[《》]", " ", title)
        cleaned = re.sub(r"(2024年|2023年|2022年|第一季度|一季度|年度|季度|报告)", " ", cleaned)
        cleaned = normalize_ws(cleaned)
        return cleaned or title

    @staticmethod
    def _item_rank_score(item: Dict[str, Any]) -> float:
        try:
            score = float(item.get("confidence", 0.0))
        except Exception:
            score = 0.0
        provenance = normalize_ws(item.get("provenance_strength", "")).lower()
        if provenance == "strong":
            score += 0.08
        elif provenance == "medium":
            score += 0.04
        elif provenance == "weak":
            score -= 0.05
        if item.get("disambiguation_needed"):
            score -= 0.12
        relation_context = item.get("source_relation_context", []) or []
        relation_types = {
            normalize_ws(relation.get("relation_type", ""))
            for relation in relation_context
            if isinstance(relation, dict)
        }
        if relation_types & {"supports", "disambiguates", "same_company_metric", "same_case", "same_target_paper"}:
            score += 0.06
        if relation_types & {"conflicts_with"}:
            score -= 0.12
        return score

    @staticmethod
    def _ranked_values(doc_results: List[Dict[str, Any]]) -> List[str]:
        values: List[str] = []
        for item in sorted(
            AnswerWriter._flatten_items(doc_results),
            key=AnswerWriter._item_rank_score,
            reverse=True,
        ):
            normalized = normalize_ws(item.get("normalized_value", "")) or normalize_ws(item.get("value", ""))
            if not normalized:
                continue
            if AnswerWriter._is_doc_placeholder(normalized):
                doc_title = AnswerWriter._doc_title_answer(str(item.get("doc_title", "")))
                if doc_title:
                    normalized = doc_title
            if normalized and normalized not in values:
                values.append(normalized)
        return values

    @staticmethod
    def _canonical_relation_key(key: str) -> str:
        lowered = normalize_ws(key).casefold()
        if "citation" in lowered or "cited" in lowered or "被引用" in lowered:
            return "Citation"
        if "reference" in lowered or "bibliography" in lowered or "参考" in lowered:
            return "Reference"
        return normalize_ws(key)

    @staticmethod
    def _target_paper_terms(record: Dict[str, Any]) -> List[str]:
        text = f"{record.get('question', '')}\n{record.get('instruction', '')}"
        terms: List[str] = []
        for term in re.findall(r"[\"“”'‘’]([^\"“”'‘’]{4,180})[\"“”'‘’]", str(text or "")):
            normalized = normalize_ws(term)
            if normalized and normalized.casefold() not in {"reference", "references", "citation", "citations"}:
                terms.append(normalized)
        return terms[:4]

    @staticmethod
    def _paper_relation_answer(record: Dict[str, Any], doc_results: List[Dict[str, Any]], current_answer: Any) -> Any:
        if not isinstance(current_answer, dict):
            current_answer = {}
        target_terms = AnswerWriter._target_paper_terms(record)
        grouped: Dict[str, List[str]] = {}

        for key, value in current_answer.items():
            canonical_key = AnswerWriter._canonical_relation_key(str(key))
            normalized_values = normalize_mapping_values(value)
            if canonical_key:
                grouped.setdefault(canonical_key, [])
                for entry in normalized_values:
                    if entry not in grouped[canonical_key]:
                        grouped[canonical_key].append(entry)

        for item in sorted(
            AnswerWriter._flatten_items(doc_results),
            key=AnswerWriter._item_rank_score,
            reverse=True,
        ):
            key = AnswerWriter._canonical_relation_key(str(item.get("answer_key", "")))
            if key not in {"Reference", "Citation"}:
                continue
            value = item.get("normalized_value", item.get("value", ""))
            evidence = f"{item.get('evidence_text', '')} {value}"
            source_title = str(item.get("doc_title", ""))
            source_is_target = any(term.casefold() in source_title.casefold() for term in target_terms)
            evidence_mentions_target = any(term.casefold() in str(evidence).casefold() for term in target_terms)
            if key == "Reference" and target_terms and evidence_mentions_target and not source_is_target:
                key = "Citation"
            elif key == "Citation" and target_terms and source_is_target:
                key = "Reference"
            grouped.setdefault(key, [])
            for entry in normalize_mapping_values(value):
                if entry not in grouped[key]:
                    grouped[key].append(entry)

        return {key: values for key, values in grouped.items() if values}

    @staticmethod
    def _guarded_answer(record: Dict[str, Any], doc_results: List[Dict[str, Any]], final_answer: Any) -> Any:
        topology = answer_topology_for_record(record)
        record_type = str(record.get("type", "")).strip().lower()
        if topology == "str" and record_type == "financial":
            answer_text = normalize_ws(final_answer)
            if AnswerWriter._is_doc_placeholder(answer_text):
                ranked_values = [value for value in AnswerWriter._ranked_values(doc_results) if not AnswerWriter._is_doc_placeholder(value)]
                return ranked_values[0] if ranked_values else ""
        if topology == "dict" and record_type == "paper":
            normalized = AnswerWriter._paper_relation_answer(record, doc_results, final_answer)
            return normalized if normalized else final_answer
        if topology == "dict" and isinstance(final_answer, dict):
            cleaned: Dict[str, List[str]] = {}
            for key, value in final_answer.items():
                key_text = normalize_ws(key)
                values = normalize_mapping_values(value)
                if key_text and values:
                    cleaned[key_text] = values
            return cleaned
        return final_answer

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
        flattened = AnswerWriter._flatten_items(doc_results)

        if topology == "str":
            values = []
            ranked_items = sorted(flattened, key=AnswerWriter._item_rank_score, reverse=True)
            for item in ranked_items:
                normalized = normalize_ws(item.get("normalized_value", "")) or normalize_ws(item.get("value", ""))
                if AnswerWriter._is_doc_placeholder(normalized):
                    normalized = AnswerWriter._doc_title_answer(str(item.get("doc_title", "")))
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
            for item in sorted(flattened, key=AnswerWriter._item_rank_score, reverse=True):
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
        for item in sorted(flattened, key=AnswerWriter._item_rank_score, reverse=True):
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
        final_answer = self._guarded_answer(record, doc_results, final_answer)
        result = {
            "topology": topology,
            "final_answer": final_answer,
            "raw_text": raw_text,
            "evidence_item_count": sum(len(result.get("items", [])) for result in doc_results),
        }
        write_json(cache_path, result)
        return result
