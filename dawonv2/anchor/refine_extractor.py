from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backend import QwenLocalClient
from .common import (
    answer_topology_for_record,
    compact_text,
    current_query_from_record,
    extract_json_payload,
    extract_tag_content,
    normalize_ws,
    read_json,
    task_mode_for_level,
    write_json,
)
from .search_r1 import SearchR1


class RefineExtractor:
    def __init__(
        self,
        llm: QwenLocalClient,
        search_agent: Optional[SearchR1] = None,
        prompt_dir: Optional[Path] = None,
        max_rounds: int = 3,
    ) -> None:
        self.llm = llm
        self.search_agent = search_agent
        self.max_rounds = max_rounds
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parent / "prompts"
        self.system_prompt = (self.prompt_dir / "refine_system.txt").read_text(encoding="utf-8").strip()
        self.user_prompt = (self.prompt_dir / "refine_user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _answer_schema_hint(record: Dict[str, Any]) -> str:
        topology = answer_topology_for_record(record)
        record_type = str(record.get("type", "")).strip().lower()
        level = int(record.get("level", 0) or 0)
        if topology == "str":
            if record_type == "financial" and level == 2:
                return "Extract per-document comparable values or matching company titles."
            if record_type == "legal" and level == 2:
                return "Extract titles of documents that match the asked legal cause."
            if record_type == "financial" and level == 4:
                return "Extract yearly values and trend evidence relevant to the asked metric."
            return "Extract the direct answer span for this document."
        if topology == "dict":
            if record_type == "paper":
                return "Extract local citation/reference relations only among provided documents."
            if record_type == "legal" and level == 4:
                return "Map document titles to the asked 判决结果 labels when supported by evidence."
            return "Extract category-to-document assignments supported by this document."
        return "Extract ordered local relations or edge facts that can help build the final chain."

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    @staticmethod
    def _normalize_items(
        raw_items: Any,
        *,
        doc_id: str,
        fallback_anchor_id: str,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            return items
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            answer_key = normalize_ws(item.get("answer_key", ""))
            value = item.get("value", "")
            if not answer_key and (value == "" or value is None or value == [] or value == {}):
                continue
            evidence_text = normalize_ws(item.get("evidence_text", ""))
            normalized_value = item.get("normalized_value", value)
            try:
                confidence = float(item.get("confidence", 0.5))
            except Exception:
                confidence = 0.5
            items.append(
                {
                    "answer_key": answer_key or "candidate",
                    "value": value,
                    "normalized_value": normalized_value,
                    "evidence_text": evidence_text,
                    "source_doc_id": doc_id,
                    "source_anchor_id": normalize_ws(item.get("source_anchor_id", "")) or fallback_anchor_id,
                    "confidence": max(0.0, min(confidence, 1.0)),
                }
            )
        return items

    @staticmethod
    def _dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            key = (
                normalize_ws(item.get("answer_key", "")),
                json.dumps(item.get("normalized_value", ""), ensure_ascii=False, sort_keys=True),
                normalize_ws(item.get("source_anchor_id", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _value_text(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return normalize_ws(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return normalize_ws(value)

    @staticmethod
    def _looks_like_doc_placeholder(value: Any, doc_titles: List[str]) -> bool:
        text = RefineExtractor._value_text(value)
        if not text:
            return False
        if re.fullmatch(r"DOC\d+", text, flags=re.IGNORECASE):
            return True
        compact_titles = {normalize_ws(title) for title in doc_titles if normalize_ws(title)}
        return text in compact_titles

    @staticmethod
    def _must_find_gaps(
        *,
        record: Dict[str, Any],
        items: List[Dict[str, Any]],
        must_find: List[str],
    ) -> List[str]:
        if not must_find:
            return []
        topology = answer_topology_for_record(record)
        record_type = str(record.get("type", "")).strip().lower()
        item_text = normalize_ws(json.dumps(items, ensure_ascii=False)).casefold()
        gaps: List[str] = []

        meaningful_items = [
            item
            for item in items
            if normalize_ws(item.get("answer_key", "")) != "candidate"
            or normalize_ws(item.get("normalized_value", "")) not in {"", "DOC1", "DOC2", "DOC3"}
        ]
        if not meaningful_items:
            return list(must_find)

        for target in must_find:
            target_norm = normalize_ws(target)
            target_lower = target_norm.casefold()
            if not target_norm:
                continue
            if "citation" in target_lower or "reference" in target_lower:
                if not re.search(r"(reference|citation|引用|参考)", item_text):
                    gaps.append(target_norm)
            elif "numeric" in target_lower or "table" in target_lower or "数" in target_norm or "表" in target_norm:
                if not re.search(r"\d", item_text):
                    gaps.append(target_norm)
            elif "category" in target_lower or "labels" in target_lower or "分类" in target_norm:
                if topology == "dict" and not any(normalize_ws(item.get("answer_key", "")) not in {"", "candidate"} for item in items):
                    gaps.append(target_norm)
            elif record_type == "paper" and topology == "dict":
                if not any(normalize_ws(item.get("answer_key", "")).casefold() in {"reference", "citation"} for item in items):
                    gaps.append(target_norm)
        return gaps

    def _coverage_hints(
        self,
        *,
        record: Dict[str, Any],
        doc_payload: Dict[str, Any],
        selected_anchor_ids: List[str],
        items: List[Dict[str, Any]],
        must_find: List[str],
    ) -> Dict[str, Any]:
        doc_titles = [normalize_ws(doc_payload.get("doc_title", ""))]
        anchor_by_id = {anchor["anchor_id"]: anchor for anchor in doc_payload.get("anchors", [])}
        known_answer_keys: List[str] = []
        known_value_signatures: List[str] = []
        low_confidence_keys: List[str] = []
        generic_item_count = 0
        doc_placeholder_values: List[str] = []

        for item in items:
            key = normalize_ws(item.get("answer_key", ""))
            if key and key not in known_answer_keys:
                known_answer_keys.append(key)
            value_text = self._value_text(item.get("normalized_value", item.get("value", "")))
            if value_text and value_text not in known_value_signatures:
                known_value_signatures.append(value_text)
            try:
                confidence = float(item.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            if confidence < 0.45 and key and key not in low_confidence_keys:
                low_confidence_keys.append(key)
            if key == "candidate" or confidence < 0.35:
                generic_item_count += 1
            if self._looks_like_doc_placeholder(item.get("normalized_value", item.get("value", "")), doc_titles):
                if value_text and value_text not in doc_placeholder_values:
                    doc_placeholder_values.append(value_text)

        missing_answer_keys: List[str] = []
        topology = answer_topology_for_record(record)
        record_type = str(record.get("type", "")).strip().lower()
        question_text = f"{record.get('question', '')}\n{record.get('instruction', '')}"
        if topology == "dict" and not known_answer_keys:
            if record_type == "paper":
                if re.search(r"(reference|参考文献)", question_text, flags=re.IGNORECASE):
                    missing_answer_keys.append("Reference")
                if re.search(r"(citation|被引用|引用)", question_text, flags=re.IGNORECASE):
                    missing_answer_keys.append("Citation")
            missing_answer_keys.extend(item for item in must_find if item not in missing_answer_keys)
        elif topology == "str" and (not items or doc_placeholder_values):
            missing_answer_keys.extend(item for item in must_find if item not in missing_answer_keys)

        visited_anchor_texts = []
        for anchor_id in selected_anchor_ids:
            anchor = anchor_by_id.get(anchor_id)
            if anchor:
                visited_anchor_texts.append(
                    f"{anchor.get('section_path', '')} {anchor.get('summary', '')} {compact_text(anchor.get('text', ''), limit=500)}"
                )

        missing_must_find = self._must_find_gaps(record=record, items=items, must_find=must_find)
        return {
            "answer_topology": topology,
            "answer_schema_hint": self._answer_schema_hint(record),
            "known_answer_keys": known_answer_keys,
            "known_value_signatures": known_value_signatures[:12],
            "missing_answer_keys": missing_answer_keys[:12],
            "selected_anchor_ids": list(selected_anchor_ids),
            "low_confidence_keys": low_confidence_keys[:12],
            "generic_item_count": generic_item_count,
            "doc_placeholder_values": doc_placeholder_values[:8],
            "needs_non_doc_placeholder": bool(doc_placeholder_values),
            "needs_more_evidence": not items or bool(missing_must_find) or bool(low_confidence_keys),
            "must_find": must_find,
            "missing_must_find": missing_must_find[:12],
            "visited_anchor_texts": visited_anchor_texts[-6:],
        }

    @staticmethod
    def _is_sufficient(coverage_hints: Dict[str, Any]) -> bool:
        if coverage_hints.get("missing_must_find"):
            return False
        if coverage_hints.get("needs_non_doc_placeholder"):
            return False
        return True

    @staticmethod
    def _ordered_anchor_ids(doc_payload: Dict[str, Any], initial_ids: List[str]) -> List[str]:
        ordered_doc_ids = [anchor["anchor_id"] for anchor in doc_payload.get("anchors", [])]
        merged: List[str] = []
        seen = set()
        for anchor_id in initial_ids + ordered_doc_ids:
            if anchor_id in seen:
                continue
            if anchor_id not in ordered_doc_ids:
                continue
            seen.add(anchor_id)
            merged.append(anchor_id)
        return merged

    @staticmethod
    def _info_payload(anchor: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "anchor_id": anchor["anchor_id"],
            "anchor_type": anchor.get("anchor_type", ""),
            "section_path": anchor.get("section_path", ""),
            "summary": anchor.get("summary", ""),
            "text": anchor.get("text", ""),
        }
        if anchor.get("source_window_anchor_ids"):
            payload["source_window_anchor_ids"] = anchor.get("source_window_anchor_ids", [])
            payload["primary_anchor_id"] = anchor.get("primary_anchor_id", anchor["anchor_id"])
        return payload

    @staticmethod
    def _expanded_anchor_window(
        *,
        doc_payload: Dict[str, Any],
        anchor: Dict[str, Any],
        max_neighbors: int = 2,
        char_limit: int = 5200,
    ) -> Dict[str, Any]:
        anchors = sorted(doc_payload.get("anchors", []), key=lambda item: int(item.get("order", 0)))
        anchor_ids = [item["anchor_id"] for item in anchors]
        if anchor["anchor_id"] not in anchor_ids:
            return anchor
        center = anchor_ids.index(anchor["anchor_id"])
        section_path = normalize_ws(anchor.get("section_path", ""))

        selected: List[Dict[str, Any]] = [anchor]
        selected_ids = {anchor["anchor_id"]}

        candidate_indices: List[int] = []
        for distance in range(1, max_neighbors + 1):
            for idx in (center - distance, center + distance):
                if 0 <= idx < len(anchors):
                    candidate_indices.append(idx)
        same_section = [
            idx
            for idx in candidate_indices
            if normalize_ws(anchors[idx].get("section_path", "")) == section_path
        ]
        fallback_neighbors = [idx for idx in candidate_indices if idx not in same_section]

        for idx in same_section + fallback_neighbors:
            candidate = anchors[idx]
            if candidate["anchor_id"] in selected_ids:
                continue
            projected = selected + [candidate]
            projected_text = "\n\n".join(str(item.get("text", "")) for item in projected)
            if len(projected_text) > char_limit:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["anchor_id"])

        selected.sort(key=lambda item: int(item.get("order", 0)))
        if len(selected) == 1:
            return anchor

        expanded = dict(anchor)
        expanded["primary_anchor_id"] = anchor["anchor_id"]
        expanded["source_window_anchor_ids"] = [item["anchor_id"] for item in selected]
        expanded["text"] = "\n\n".join(
            "\n".join(
                [
                    f"[{item['anchor_id']} | {normalize_ws(item.get('section_path', '')) or '(root)'}]",
                    str(item.get("text", "")),
                ]
            )
            for item in selected
        )
        expanded["summary"] = (
            f"Expanded same-section read window around {anchor['anchor_id']}: "
            f"{compact_text(anchor.get('summary', ''), limit=180)}"
        )
        return expanded

    def _extract_from_chunk(
        self,
        *,
        record: Dict[str, Any],
        doc_payload: Dict[str, Any],
        anchor: Dict[str, Any],
        current_query: str,
        prior_trace: str,
        known_items: List[Dict[str, Any]],
        round_index: int,
    ) -> Dict[str, Any]:
        selected_chunk = self._info_payload(anchor)
        prompt = self.user_prompt.format(
            record_type=record.get("type"),
            level=record.get("level"),
            task_mode=task_mode_for_level(int(record.get("level", 0) or 0)),
            answer_topology=answer_topology_for_record(record),
            answer_schema_hint=self._answer_schema_hint(record),
            round_index=round_index,
            current_query=current_query or "(empty)",
            question=str(record.get("question", "")).strip() or "(empty)",
            instruction=str(record.get("instruction", "")).strip(),
            doc_title=doc_payload["doc_title"],
            previous_trace=prior_trace.strip() or "(empty)",
            known_items_json=json.dumps(known_items[-6:], ensure_ascii=False, indent=2),
            selected_chunk_json=json.dumps(selected_chunk, ensure_ascii=False, indent=2),
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            max_output_tokens=1200,
            metadata={
                "module": "refine_extractor",
                "doc_title": doc_payload["doc_title"],
                "round_index": round_index,
            },
        )

        think_text = normalize_ws(extract_tag_content(raw_text, "think") or "")
        retrieve_text = extract_tag_content(raw_text, "retrieve") or ""
        payload = extract_json_payload(retrieve_text or "")

        raw_items: Any = []
        enough = False
        followup_query = ""
        if isinstance(payload, dict):
            raw_items = payload.get("items", [])
            enough = self._coerce_bool(payload.get("enough", False))
            followup_query = normalize_ws(payload.get("followup_query", ""))
        elif isinstance(payload, list):
            raw_items = payload

        items = self._normalize_items(
            raw_items,
            doc_id=doc_payload["doc_id"],
            fallback_anchor_id=anchor["anchor_id"],
        )
        if not followup_query and not enough and not items:
            followup_query = current_query

        retrieve_payload = {
            "items": items,
            "enough": enough,
            "followup_query": followup_query,
        }
        return {
            "think": think_text,
            "raw_text": raw_text,
            "retrieve_payload": retrieve_payload,
            "items": items,
            "enough": enough,
            "followup_query": followup_query,
        }

    def run(
        self,
        *,
        record: Dict[str, Any],
        doc_payload: Dict[str, Any],
        search_output: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / f"{doc_payload['doc_id']}_refine.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        anchor_by_id = {anchor["anchor_id"]: anchor for anchor in doc_payload.get("anchors", [])}
        remaining_anchor_ids = self._ordered_anchor_ids(
            doc_payload,
            search_output.get("all_ranked_anchor_ids", []) or search_output.get("ranked_anchor_ids", []),
        )
        if not remaining_anchor_ids:
            remaining_anchor_ids = [anchor["anchor_id"] for anchor in doc_payload.get("anchors", [])]

        current_query = search_output.get("current_query", "") or current_query_from_record(
            str(record.get("question", "")).strip(),
            str(record.get("instruction", "")).strip(),
        )
        must_find = list(search_output.get("must_find", []))
        aggregated_items: List[Dict[str, Any]] = []
        selected_anchor_ids: List[str] = []
        rounds: List[Dict[str, Any]] = []
        trace_lines: List[str] = []

        for round_index in range(1, self.max_rounds + 1):
            if not remaining_anchor_ids:
                break

            coverage_hints = self._coverage_hints(
                record=record,
                doc_payload=doc_payload,
                selected_anchor_ids=selected_anchor_ids,
                items=aggregated_items,
                must_find=must_find,
            )

            if round_index == 1:
                decision = {
                    "think": search_output.get("think", ""),
                    "current_query": current_query,
                    "must_find": must_find,
                    "next_anchor_id": search_output.get("next_anchor_id", "") or remaining_anchor_ids[0],
                    "candidate_anchor_ids": search_output.get("candidate_anchor_ids", []),
                    "ranked_anchor_ids": search_output.get("ranked_anchor_ids", []),
                    "utility_hits": search_output.get("utility_hits", {}),
                    "utility_score": search_output.get("utility_score", {}),
                    "score_breakdown": search_output.get("score_breakdown", {}),
                    "raw_plan_text": search_output.get("raw_plan_text", ""),
                }
            elif self.search_agent is not None:
                decision = self.search_agent.choose_next_anchor(
                    record=record,
                    doc_payload=doc_payload,
                    remaining_anchor_ids=remaining_anchor_ids,
                    prior_trace="\n".join(trace_lines),
                    known_items=aggregated_items,
                    current_query_override=current_query,
                    must_find_override=must_find,
                    coverage_hints=coverage_hints,
                    avoid_anchor_ids=selected_anchor_ids,
                    round_index=round_index,
                )
            else:
                decision = {
                    "think": f"I still need more evidence from {doc_payload['doc_title']}.",
                    "current_query": current_query,
                    "must_find": must_find,
                    "next_anchor_id": remaining_anchor_ids[0],
                    "candidate_anchor_ids": remaining_anchor_ids[:4],
                    "ranked_anchor_ids": remaining_anchor_ids[:4],
                    "coverage_hints": coverage_hints,
                    "avoid_anchor_ids": selected_anchor_ids,
                    "raw_plan_text": "",
                }

            next_anchor_id = normalize_ws(decision.get("next_anchor_id", ""))
            if next_anchor_id not in anchor_by_id:
                next_anchor_id = remaining_anchor_ids[0]
            anchor = anchor_by_id[next_anchor_id]
            remaining_anchor_ids = [anchor_id for anchor_id in remaining_anchor_ids if anchor_id != next_anchor_id]
            selected_anchor_ids.append(next_anchor_id)
            expanded_anchor = self._expanded_anchor_window(doc_payload=doc_payload, anchor=anchor)

            plan_think = normalize_ws(decision.get("think", ""))
            current_query = normalize_ws(decision.get("current_query", "")) or current_query
            if decision.get("must_find"):
                must_find = [normalize_ws(item) for item in decision.get("must_find", []) if normalize_ws(item)]
            search_payload = {
                "current_query": current_query,
                "must_find": must_find,
                "next_anchor_id": next_anchor_id,
                "candidate_anchor_ids": decision.get("candidate_anchor_ids", []),
            }
            for optional_key in ("coverage_hints", "avoid_anchor_ids", "utility_hits", "utility_score", "score_breakdown"):
                if decision.get(optional_key):
                    search_payload[optional_key] = decision.get(optional_key)
            if coverage_hints:
                search_payload.setdefault("coverage_hints", coverage_hints)
            info_payload = self._info_payload(expanded_anchor)

            trace_lines.append(f"<think>{plan_think}</think>")
            trace_lines.append(f"<search>{json.dumps(search_payload, ensure_ascii=False)}</search>")
            trace_lines.append(f"<info>{json.dumps(info_payload, ensure_ascii=False)}</info>")

            extraction = self._extract_from_chunk(
                record=record,
                doc_payload=doc_payload,
                anchor=expanded_anchor,
                current_query=current_query,
                prior_trace="\n".join(trace_lines),
                known_items=aggregated_items,
                round_index=round_index,
            )
            retrieve_payload = extraction["retrieve_payload"]

            aggregated_items.extend(extraction["items"])
            aggregated_items = self._dedupe_items(aggregated_items)

            coverage_after = self._coverage_hints(
                record=record,
                doc_payload=doc_payload,
                selected_anchor_ids=selected_anchor_ids,
                items=aggregated_items,
                must_find=must_find,
            )
            if extraction["enough"] and not self._is_sufficient(coverage_after):
                extraction["enough"] = False
                retrieve_payload["enough"] = False
                retrieve_payload["sufficiency_forced_false"] = True
                retrieve_payload["sufficiency_reasons"] = coverage_after.get("missing_must_find", []) or coverage_after.get(
                    "doc_placeholder_values", []
                )
            retrieve_payload["coverage_hints"] = coverage_after

            trace_lines.append(f"<think>{extraction['think']}</think>")
            trace_lines.append(f"<retrieve>{json.dumps(retrieve_payload, ensure_ascii=False)}</retrieve>")

            rounds.append(
                {
                    "round_index": round_index,
                    "search": search_payload,
                    "anchor_id": next_anchor_id,
                    "plan_think": plan_think,
                    "info": info_payload,
                    "extract_think": extraction["think"],
                    "retrieve": retrieve_payload,
                }
            )

            if extraction["followup_query"]:
                current_query = extraction["followup_query"]
            if extraction["enough"]:
                break

        if not aggregated_items:
            topology = answer_topology_for_record(record)
            if topology == "str":
                fallback_anchor = anchor_by_id[selected_anchor_ids[0]] if selected_anchor_ids else next(iter(anchor_by_id.values()), {})
                aggregated_items = [
                    {
                        "answer_key": "candidate",
                        "value": compact_text(fallback_anchor.get("text", ""), limit=160),
                        "normalized_value": compact_text(fallback_anchor.get("text", ""), limit=160),
                        "evidence_text": compact_text(fallback_anchor.get("text", ""), limit=200),
                        "source_doc_id": doc_payload["doc_id"],
                        "source_anchor_id": fallback_anchor.get("anchor_id", ""),
                        "confidence": 0.15,
                    }
                ]

        payload_out = {
            "doc_id": doc_payload["doc_id"],
            "doc_title": doc_payload["doc_title"],
            "selected_anchor_ids": selected_anchor_ids,
            "final_current_query": current_query,
            "rounds": rounds,
            "items": aggregated_items,
            "trace": {
                "round_count": len(rounds),
                "tagged_trace": "\n".join(trace_lines),
            },
            "tagged_trace": "\n".join(trace_lines),
        }
        write_json(cache_path, payload_out)
        return payload_out
