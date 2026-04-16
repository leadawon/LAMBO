"""DocRefineAgent — stateful multi-turn per-document evidence extraction.

This v2 version keeps explicit per-document search memory so the agent can
remember opened anchors, missing slots, and whether to continue locally or jump
elsewhere in the document. Anchor summaries are treated as routing hints only;
only opened raw anchor text is valid evidence.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..backend import GeminiClient, QwenLocalClient
from ..common import (
    compact_text,
    extract_json_payload,
    extract_tag_content,
    normalize_ws,
    quoted_terms,
    read_json,
    tokenize_query,
    write_json,
)


@dataclass
class SearchState:
    current_query: str
    must_find: List[str]
    seen_anchor_ids: List[str] = field(default_factory=list)
    known_items: List[Dict[str, Any]] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)
    failed_queries: List[str] = field(default_factory=list)
    last_anchor_id: str = ""
    last_action: str = "inspect_anchor"
    round_index: int = 1
    local_frontier_anchor_ids: List[str] = field(default_factory=list)
    global_frontier_anchor_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocRefineAgent:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient],
        prompt_dir: Optional[Path] = None,
        max_rounds: int = 6,
        max_catalog_anchors: int = 10,
        local_window_size: int = 4,
    ) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        self.max_catalog_anchors = max_catalog_anchors
        self.local_window_size = local_window_size
        pdir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts" / "doc_refine"
        self.system_prompt = (pdir / "system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (pdir / "user.txt").read_text(encoding="utf-8").strip()

    @staticmethod
    def _format_doc_map(anchors: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for anchor in anchors:
            entities = ", ".join(anchor.get("key_entities") or [])
            summary = compact_text(anchor.get("summary", ""), limit=220)
            lines.append(
                f"- {anchor['anchor_id']} | {anchor.get('anchor_title', '')}"
                + (f"\n    key_entities: {entities}" if entities else "")
                + (f"\n    summary: {summary}" if summary else "")
            )
        return "\n".join(lines) if lines else "(no candidate anchors available)"

    @staticmethod
    def _format_evidence(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "(none yet)"
        rows: List[str] = []
        for i, item in enumerate(items, start=1):
            src = item.get("source_anchor_id", "")
            fact = compact_text(item.get("fact", ""), limit=220)
            ents = ", ".join(item.get("entities") or [])
            rows.append(f"{i}. [{src}] {fact}" + (f"  (entities: {ents})" if ents else ""))
        return "\n".join(rows)

    @staticmethod
    def _state_summary(state: SearchState) -> str:
        payload = {
            "current_query": state.current_query,
            "must_find": state.must_find,
            "seen_anchor_ids": state.seen_anchor_ids[-6:],
            "missing_slots": state.missing_slots,
            "last_anchor_id": state.last_anchor_id,
            "last_action": state.last_action,
            "round_index": state.round_index,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_target_terms(query: str, instruction: str, current_query: str) -> List[str]:
        text = "\n".join([query, instruction, current_query])
        quoted = re.findall(r"[‘'\"]([^‘'\"]{2,80})[’'\"]", text)
        bracketed = re.findall(r"《([^》]{2,80})》", text)
        phrases = re.findall(r"[\u4e00-\u9fffA-Za-z0-9()（）·\-]{3,40}", text)
        banned = {
            "question",
            "instruction",
            "please answer",
            "based on",
            "according to",
            "your answer",
            "question answer",
            "以上",
            "根据",
            "请回答",
            "作答",
        }
        deduped: List[str] = []
        for item in quoted + bracketed + phrases:
            normalized = normalize_ws(item)
            if len(normalized) < 3:
                continue
            if normalized.casefold() in banned:
                continue
            if normalized not in deduped:
                deduped.append(normalized)
        deduped.sort(key=len, reverse=True)
        return deduped[:8]

    @staticmethod
    def _row_header(text: str) -> str:
        stripped = normalize_ws(text)
        if not stripped:
            return ""
        if stripped.startswith("|"):
            parts = [normalize_ws(part) for part in stripped.split("|") if normalize_ws(part)]
            return parts[0] if parts else ""
        return stripped.split("\n", 1)[0][:80]

    @staticmethod
    def _is_header_like(anchor: Dict[str, Any]) -> bool:
        text = normalize_ws(str(anchor.get("text", "")))
        if not text:
            return True
        if text.startswith("|"):
            parts = [normalize_ws(part) for part in text.split("|") if normalize_ws(part)]
            if len(parts) <= 2:
                return True
            if all(part in {"-", "--", "---"} for part in parts[1:]):
                return True
            if parts[0].endswith("：") and all(part in {"-", "--", "---"} for part in parts[1:]):
                return True
            if parts[0] in {"项目", "流动资产：", "非流动资产：", "流动负债：", "非流动负债：", "所有者权益："}:
                return True
        return False

    def _initial_must_find(self, *, query: str, instruction: str) -> List[str]:
        text = f"{query}\n{instruction}".casefold()
        needs: List[str] = []
        if re.search(r"(最高|最低|largest|smallest|highest|lowest|比较|排序|rank)", text):
            needs.append("comparable values for each relevant document")
        if re.search(r"(引用|参考文献|citation|reference|bibliography)", text):
            needs.append("citation or reference evidence")
        if re.search(r"(判决|裁定|案由|法院|verdict|disposition|cause)", text):
            needs.append("case cause, ruling, or disposition evidence")
        if re.search(r"(资产|利润|收入|现金|financial|table|数值|金额|比例|trend)", text):
            needs.append("numeric or table evidence")
        if not needs:
            needs.append("direct answer evidence")
        return needs

    def _anchor_neighbors(self, anchors: List[Dict[str, Any]], anchor_id: str) -> List[str]:
        index_by_id = {anchor["anchor_id"]: idx for idx, anchor in enumerate(anchors)}
        if anchor_id not in index_by_id:
            return []
        idx = index_by_id[anchor_id]
        start = max(0, idx - self.local_window_size)
        end = min(len(anchors), idx + self.local_window_size + 1)
        local_ids: List[str] = []
        anchor = anchors[idx]
        anchor_title = normalize_ws(anchor.get("anchor_title", ""))
        for neighbor in anchors[start:end]:
            if neighbor["anchor_id"] not in local_ids:
                local_ids.append(neighbor["anchor_id"])
        for linked_id in [anchor.get("prev_anchor_id", ""), anchor.get("next_anchor_id", "")]:
            linked_id = normalize_ws(linked_id)
            if linked_id and linked_id not in local_ids:
                local_ids.append(linked_id)
        if anchor_title:
            for neighbor in anchors:
                if neighbor["anchor_id"] in local_ids:
                    continue
                if normalize_ws(neighbor.get("anchor_title", "")) == anchor_title:
                    local_ids.append(neighbor["anchor_id"])
        return local_ids

    def _summarize_missing_slots(self, *, query: str, instruction: str, state: SearchState) -> List[str]:
        target_terms = self._extract_target_terms(query, instruction, state.current_query)
        evidence_text = " ".join(
            filter(
                None,
                [
                    normalize_ws(item.get("fact", "")) + " " + " ".join(item.get("entities") or [])
                    for item in state.known_items
                ],
            )
        ).casefold()
        missing: List[str] = []
        for term in target_terms:
            lowered = term.casefold()
            if lowered and lowered not in evidence_text:
                missing.append(term)
        if not missing:
            missing.extend(state.must_find[:2])
        deduped: List[str] = []
        for item in missing:
            normalized = normalize_ws(item)
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped[:4]

    def initialize_state(self, *, query: str, instruction: str, anchors: List[Dict[str, Any]]) -> SearchState:
        state = SearchState(
            current_query=normalize_ws(query),
            must_find=self._initial_must_find(query=query, instruction=instruction),
            global_frontier_anchor_ids=[anchor["anchor_id"] for anchor in anchors],
        )
        state.missing_slots = self._summarize_missing_slots(query=query, instruction=instruction, state=state)
        return state

    def _score_anchor(
        self,
        *,
        query: str,
        instruction: str,
        doc_title: str,
        anchor: Dict[str, Any],
        state: SearchState,
    ) -> Tuple[float, List[str]]:
        anchor_title = str(anchor.get("anchor_title", ""))
        summary = str(anchor.get("summary", ""))
        entities = " ".join(anchor.get("key_entities") or [])
        text_preview = compact_text(str(anchor.get("text", "")), limit=320)
        row_header = self._row_header(text_preview)
        combined = f"{doc_title} {anchor_title} {summary} {entities} {text_preview} {row_header}"
        combined_lower = combined.casefold()

        query_tokens = tokenize_query(state.current_query or query)
        instruction_tokens = tokenize_query(instruction)
        quote_tokens = quoted_terms(f"{query}\n{instruction}")
        target_terms = self._extract_target_terms(query, instruction, state.current_query)

        score = 0.0
        reasons: List[str] = []

        overlap = sum(1 for token in query_tokens if token in combined_lower)
        if overlap:
            score += overlap * 7
            reasons.append(f"query_overlap={overlap}")

        instruction_overlap = sum(1 for token in instruction_tokens[:12] if token in combined_lower)
        if instruction_overlap:
            score += min(instruction_overlap, 4) * 2.5
            reasons.append(f"instruction_overlap={instruction_overlap}")

        quote_hits = sum(1 for token in quote_tokens if token.casefold() in combined_lower)
        if quote_hits:
            score += quote_hits * 10
            reasons.append(f"quoted_hit={quote_hits}")

        exact_target_hits = 0
        for term in target_terms:
            lowered = term.casefold()
            if lowered and lowered in combined_lower:
                exact_target_hits += 1
                if lowered == row_header.casefold():
                    score += 24
                    reasons.append(f"row_header_exact={term[:24]}")
                else:
                    score += 16
                    reasons.append(f"target_exact={term[:24]}")
        if exact_target_hits:
            score += exact_target_hits * 3

        if any(re.search(r"\d", token) for token in query_tokens) and re.search(r"\d", combined):
            score += 5
            reasons.append("numeric_match")

        if re.search(r"(citation|reference|bibliography|et al\.|\[[0-9]+\]|参考文献|引用)", combined_lower):
            score += 8
            reasons.append("citation_cue")

        if re.search(r"(案由|判决|裁定|撤销|驳回|罪|纠纷|行政|法院|ruling|verdict|disposition)", combined_lower):
            score += 6
            reasons.append("legal_cue")

        if re.search(r"(元|%|revenue|profit|cash|资产|利润|收入|应收|负债|金额|同比|环比)", combined_lower):
            score += 6
            reasons.append("financial_cue")

        if self._is_header_like(anchor):
            score -= 18
            reasons.append("header_penalty")

        if anchor["anchor_id"] in state.seen_anchor_ids:
            score -= 20
            reasons.append("seen_penalty")
        if anchor["anchor_id"] in state.local_frontier_anchor_ids:
            score += 9
            reasons.append("local_frontier_boost")
        if state.missing_slots:
            slot_hits = sum(1 for slot in state.missing_slots if slot.casefold() in combined_lower)
            if slot_hits:
                score += slot_hits * 8
                reasons.append(f"missing_slot_hit={slot_hits}")

        order = int(anchor.get("order", 1) or 1)
        position_bonus = max(0.0, 2.5 - math.log(order + 1))
        score += position_bonus
        reasons.append(f"position_bonus={position_bonus:.2f}")
        return score, reasons

    def _score_all_anchors(
        self,
        *,
        query: str,
        instruction: str,
        doc_title: str,
        anchors: List[Dict[str, Any]],
        remaining_anchor_ids: List[str],
        state: SearchState,
    ) -> List[Tuple[float, Dict[str, Any], List[str]]]:
        allowed = set(remaining_anchor_ids)
        scored: List[Tuple[float, Dict[str, Any], List[str]]] = []
        for anchor in anchors:
            if anchor["anchor_id"] not in allowed:
                continue
            score, reasons = self._score_anchor(
                query=query,
                instruction=instruction,
                doc_title=doc_title,
                anchor=anchor,
                state=state,
            )
            scored.append((score, anchor, reasons))
        scored.sort(key=lambda item: (item[0], -int(item[1].get("order", 0))), reverse=True)
        return scored

    def _candidate_anchor_ids(
        self,
        *,
        shortlist: List[Tuple[float, Dict[str, Any], List[str]]],
        state: SearchState,
    ) -> List[str]:
        ordered: List[str] = []
        for anchor_id in state.local_frontier_anchor_ids + [anchor["anchor_id"] for _, anchor, _ in shortlist]:
            if anchor_id and anchor_id not in ordered:
                ordered.append(anchor_id)
        return ordered[:4]

    def _next_query_if_fail(self, *, query: str, instruction: str, state: SearchState) -> str:
        if state.missing_slots:
            slot_hint = ", ".join(state.missing_slots[:2])
            return normalize_ws(f"{query} Focus on evidence about: {slot_hint}")
        target_terms = self._extract_target_terms(query, instruction, state.current_query)
        if target_terms:
            return normalize_ws(f"{query} Focus on: {target_terms[0]}")
        return state.current_query or query

    def _call_llm(
        self,
        *,
        question: str,
        instruction: str,
        doc_id: str,
        doc_title: str,
        doc_map_text: str,
        state: SearchState,
        accumulated_trace: str,
        opened_anchors: List[str],
    ) -> str:
        trace_section = accumulated_trace.strip() or "(none yet)"
        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            doc_title=doc_title,
            doc_id=doc_id,
            search_state=self._state_summary(state),
            missing_slots=json.dumps(state.missing_slots, ensure_ascii=False),
            opened_anchors=", ".join(opened_anchors) if opened_anchors else "(none)",
            evidence_so_far=self._format_evidence(state.known_items),
            doc_map=doc_map_text,
            accumulated_trace=trace_section,
        )
        return self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=1500,
            metadata={
                "module": "doc_refine",
                "phase": "think_search",
                "doc_id": doc_id,
                "round": state.round_index,
            },
        )

    @staticmethod
    def _supported_answer(answer_text: str, opened_texts: List[str]) -> str:
        if not answer_text.strip():
            return ""
        normalized_sources = [normalize_ws(text) for text in opened_texts if normalize_ws(text)]
        if not normalized_sources:
            return ""
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", answer_text) if chunk.strip()]
        if not chunks:
            chunks = [answer_text.strip()]
        kept: List[str] = []
        seen = set()
        for chunk in chunks:
            normalized = normalize_ws(chunk)
            if len(normalized) < 8:
                continue
            if any(normalized in source for source in normalized_sources):
                if normalized not in seen:
                    kept.append(chunk.strip())
                    seen.add(normalized)
        if kept:
            return "\n\n".join(kept)
        whole = normalize_ws(answer_text)
        if len(whole) >= 8 and any(whole in source for source in normalized_sources):
            return answer_text.strip()
        return ""

    @staticmethod
    def _chunk_text_for_evidence(text: str) -> List[str]:
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", str(text or "")) if chunk.strip()]
        if chunks:
            return chunks
        return [line.strip() for line in str(text or "").splitlines() if line.strip()]

    def _fallback_evidence_from_opened(
        self,
        *,
        question: str,
        instruction: str,
        opened_anchor_payloads: List[Dict[str, Any]],
    ) -> str:
        if not opened_anchor_payloads:
            return ""

        target_terms = self._extract_target_terms(question, instruction, question)
        query_tokens = tokenize_query(question)
        instruction_tokens = tokenize_query(instruction)
        scored_chunks: List[Tuple[float, str]] = []

        for anchor in opened_anchor_payloads:
            anchor_title = normalize_ws(anchor.get("anchor_title", ""))
            for chunk in self._chunk_text_for_evidence(anchor.get("text", "")):
                normalized_chunk = normalize_ws(chunk)
                if len(normalized_chunk) < 20:
                    continue
                lowered = normalized_chunk.casefold()
                score = 0.0
                for term in target_terms:
                    if term.casefold() in lowered:
                        score += 10.0 + min(len(term) / 10.0, 4.0)
                score += sum(1.5 for token in query_tokens[:12] if token in lowered)
                score += sum(0.75 for token in instruction_tokens[:12] if token in lowered)
                if anchor_title and anchor_title.casefold() in lowered:
                    score += 2.0
                if re.search(r"\d", normalized_chunk):
                    score += 1.0
                if re.search(r"(method|model|approach|dataset|experiment|result|conclusion|方法|模型|实验|结果|结论|引用|参考文献)", lowered):
                    score += 1.5
                if score > 0:
                    scored_chunks.append((score, chunk.strip()))

        if not scored_chunks:
            fallback_chunks: List[str] = []
            for anchor in opened_anchor_payloads[:2]:
                chunk_candidates = self._chunk_text_for_evidence(anchor.get("text", ""))
                if chunk_candidates:
                    fallback_chunks.append(chunk_candidates[0].strip())
            return "\n\n".join(chunk for chunk in fallback_chunks if chunk)

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        selected: List[str] = []
        seen = set()
        for _, chunk in scored_chunks:
            normalized = normalize_ws(chunk)
            if normalized in seen:
                continue
            selected.append(chunk)
            seen.add(normalized)
            if len(selected) >= 3:
                break
        return "\n\n".join(selected)

    def _plan_step(
        self,
        *,
        question: str,
        instruction: str,
        doc_id: str,
        doc_title: str,
        anchors: List[Dict[str, Any]],
        state: SearchState,
        accumulated_trace: str,
    ) -> Dict[str, Any]:
        remaining_anchor_ids = [aid for aid in state.global_frontier_anchor_ids if aid not in state.seen_anchor_ids]
        scored = self._score_all_anchors(
            query=question,
            instruction=instruction,
            doc_title=doc_title,
            anchors=anchors,
            remaining_anchor_ids=remaining_anchor_ids,
            state=state,
        )
        shortlist = scored[: self.max_catalog_anchors]
        candidate_anchor_ids = self._candidate_anchor_ids(shortlist=shortlist, state=state)
        default_next_anchor_id = shortlist[0][1]["anchor_id"] if shortlist else ""
        action_type = "scan_neighbors" if state.local_frontier_anchor_ids else "inspect_anchor"
        next_query_if_fail = self._next_query_if_fail(query=question, instruction=instruction, state=state)

        result = {
            "think": (
                f"I should inspect one anchor from {doc_title} that best matches the remaining need: "
                f"{', '.join(state.missing_slots or state.must_find) or 'task-relevant evidence'}."
            ),
            "action": "open" if default_next_anchor_id else "stop",
            "anchor_id": default_next_anchor_id,
            "reason": "heuristic top candidate",
            "candidate_anchor_ids": candidate_anchor_ids,
            "ranked_anchor_ids": [anchor["anchor_id"] for _, anchor, _ in scored[:4]],
            "all_ranked_anchor_ids": [anchor["anchor_id"] for _, anchor, _ in scored],
            "rule_hits": [
                {
                    "anchor_id": anchor["anchor_id"],
                    "score": round(score, 3),
                    "reasons": reasons,
                    "summary": compact_text(anchor.get("summary", ""), limit=180),
                }
                for score, anchor, reasons in scored[:4]
            ],
            "raw_text": "",
            "action_type": action_type,
            "next_query_if_fail": next_query_if_fail,
            "search_state": state.to_dict(),
            "answer_text": "",
        }

        if not shortlist:
            result["action"] = "stop"
            result["reason"] = "no plausible anchors remain"
            return result

        shortlist_doc_map = [
            {
                "anchor_id": anchor["anchor_id"],
                "anchor_title": anchor.get("anchor_title", ""),
                "summary": anchor.get("summary", ""),
                "key_entities": anchor.get("key_entities", []),
            }
            for _, anchor, _ in shortlist[: min(len(shortlist), self.max_catalog_anchors)]
        ]
        raw_text = self._call_llm(
            question=state.current_query or question,
            instruction=instruction,
            doc_id=doc_id,
            doc_title=doc_title,
            doc_map_text=self._format_doc_map(shortlist_doc_map),
            state=state,
            accumulated_trace=accumulated_trace,
            opened_anchors=state.seen_anchor_ids,
        )
        result["raw_text"] = raw_text
        think_text = normalize_ws(extract_tag_content(raw_text, "think") or "")
        if think_text:
            result["think"] = think_text
        search_text = extract_tag_content(raw_text, "search") or ""
        payload = extract_json_payload(search_text) or {}
        if isinstance(payload, dict):
            action = normalize_ws(str(payload.get("action", ""))).lower()
            anchor_id = normalize_ws(str(payload.get("anchor_id", "")))
            reason = normalize_ws(str(payload.get("reason", "")))
            valid_ids = set(candidate_anchor_ids)
            if action in {"open", "stop"}:
                result["action"] = action
            if anchor_id in valid_ids:
                result["anchor_id"] = anchor_id
            if reason:
                result["reason"] = reason
        result["answer_text"] = extract_tag_content(raw_text, "answer") or ""
        return result

    def _update_state_after_open(
        self,
        *,
        query: str,
        instruction: str,
        anchors: List[Dict[str, Any]],
        state: SearchState,
        selected_anchor: Dict[str, Any],
        next_query_if_fail: str,
        remaining_anchor_ids: List[str],
    ) -> SearchState:
        anchor_id = selected_anchor["anchor_id"]
        if anchor_id not in state.seen_anchor_ids:
            state.seen_anchor_ids.append(anchor_id)
        state.last_anchor_id = anchor_id
        state.round_index += 1
        state.global_frontier_anchor_ids = list(remaining_anchor_ids)
        state.local_frontier_anchor_ids = [
            candidate_id
            for candidate_id in self._anchor_neighbors(anchors, anchor_id)
            if candidate_id in remaining_anchor_ids and candidate_id not in state.seen_anchor_ids
        ]

        known_item = {
            "fact": compact_text(selected_anchor.get("text", ""), limit=240),
            "entities": [normalize_ws(e) for e in (selected_anchor.get("key_entities") or []) if normalize_ws(e)],
            "source_anchor_id": anchor_id,
        }
        if not any(item.get("source_anchor_id") == anchor_id for item in state.known_items):
            state.known_items.append(known_item)

        previous_missing = list(state.missing_slots)
        state.missing_slots = self._summarize_missing_slots(query=query, instruction=instruction, state=state)
        if previous_missing == state.missing_slots and next_query_if_fail and next_query_if_fail != state.current_query:
            state.failed_queries.append(state.current_query)
            state.current_query = next_query_if_fail
            state.last_action = "rewrite_query"
        else:
            state.last_action = "scan_neighbors" if state.local_frontier_anchor_ids else "jump_to_section"
        return state

    def run(
        self,
        *,
        question: str,
        instruction: str,
        record: Dict[str, Any],
        doc_payload: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        del record
        doc_id = doc_payload["doc_id"]
        doc_title = doc_payload["doc_title"]
        cache_path = sample_dir / f"{doc_id}_refine.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        all_anchors = doc_payload.get("anchors", [])
        anchor_by_id = {anchor["anchor_id"]: anchor for anchor in all_anchors}
        state = self.initialize_state(query=question, instruction=instruction, anchors=all_anchors)
        opened_anchors: List[str] = []
        opened_texts: List[str] = []
        opened_anchor_payloads: List[Dict[str, Any]] = []
        accumulated_trace = ""
        rounds: List[Dict[str, Any]] = []
        final_decision: Dict[str, Any] = {}

        scan_result = "no_evidence"
        evidence_text = ""

        max_rounds = min(self.max_rounds, max(1, len(all_anchors)))
        for round_idx in range(1, max_rounds + 1):
            remaining_anchor_ids = [anchor_id for anchor_id in anchor_by_id if anchor_id not in opened_anchors]
            if not remaining_anchor_ids:
                break
            state.global_frontier_anchor_ids = list(remaining_anchor_ids)
            state.round_index = round_idx

            decision = self._plan_step(
                question=question,
                instruction=instruction,
                doc_id=doc_id,
                doc_title=doc_title,
                anchors=all_anchors,
                state=state,
                accumulated_trace=accumulated_trace,
            )
            final_decision = decision

            accumulated_trace += f"\n--- Round {round_idx} ---\n"
            accumulated_trace += f"<think>{decision['think']}</think>\n"
            accumulated_trace += (
                "<search>"
                + json.dumps(
                    {
                        "action": decision.get("action", "stop"),
                        "anchor_id": decision.get("anchor_id", ""),
                        "reason": decision.get("reason", ""),
                        "action_type": decision.get("action_type", "inspect_anchor"),
                        "next_query_if_fail": decision.get("next_query_if_fail", state.current_query),
                    },
                    ensure_ascii=False,
                )
                + "</search>\n"
            )

            if decision.get("action") == "stop":
                supported_answer = self._supported_answer(decision.get("answer_text", ""), opened_texts)
                if supported_answer:
                    evidence_text = supported_answer
                    scan_result = "evidence_found"
                    accumulated_trace += f"<answer>{supported_answer}</answer>\n"
                rounds.append(
                    {
                        "round_index": round_idx,
                        "plan": decision,
                        "anchor_id": "",
                        "opened": False,
                        "search_state": state.to_dict(),
                    }
                )
                state.last_action = "stop_if_enough"
                break

            anchor_id = normalize_ws(decision.get("anchor_id", ""))
            if anchor_id not in anchor_by_id or anchor_id in opened_anchors:
                anchor_id = remaining_anchor_ids[0]

            anchor = anchor_by_id[anchor_id]
            opened_anchors.append(anchor_id)
            opened_texts.append(str(anchor.get("text", "")))
            opened_anchor_payloads.append(anchor)
            accumulated_trace += f"<info anchor_id=\"{anchor_id}\">\n{anchor.get('text', '')}\n</info>\n"

            remaining_anchor_ids = [candidate for candidate in anchor_by_id if candidate not in opened_anchors]
            state = self._update_state_after_open(
                query=question,
                instruction=instruction,
                anchors=all_anchors,
                state=state,
                selected_anchor=anchor,
                next_query_if_fail=decision.get("next_query_if_fail", state.current_query),
                remaining_anchor_ids=remaining_anchor_ids,
            )
            rounds.append(
                {
                    "round_index": round_idx,
                    "plan": decision,
                    "anchor_id": anchor_id,
                    "opened": True,
                    "search_state": state.to_dict(),
                }
            )

        if not evidence_text:
            fallback_evidence = self._fallback_evidence_from_opened(
                question=question,
                instruction=instruction,
                opened_anchor_payloads=opened_anchor_payloads,
            )
            if fallback_evidence.strip():
                evidence_text = fallback_evidence
                scan_result = "evidence_found"

        evidence_sheet = {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "scan_result": scan_result,
            "evidence": evidence_text,
            "opened_anchors": opened_anchors,
            "rounds_used": len(opened_anchors),
            "trace": accumulated_trace,
            "rounds": rounds,
            "action_type": final_decision.get("action_type", state.last_action),
            "next_query_if_fail": final_decision.get("next_query_if_fail", state.current_query),
            "candidate_anchor_ids": final_decision.get("candidate_anchor_ids", []),
            "ranked_anchor_ids": final_decision.get("ranked_anchor_ids", []),
            "all_ranked_anchor_ids": final_decision.get("all_ranked_anchor_ids", []),
            "rule_hits": final_decision.get("rule_hits", []),
            "search_state": state.to_dict(),
        }
        write_json(cache_path, evidence_sheet)
        return evidence_sheet
