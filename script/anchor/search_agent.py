"""Search Agent — per-document stateful multi-turn search over anchors.

The new version keeps an explicit search state so each round can remember:
  * which anchors were already opened,
  * what evidence has already been extracted,
  * what slots still seem missing,
  * whether to continue locally around the current anchor or jump globally.

This ports the multi-turn behavior we previously added in LAMBO_before into the
newer agentic pipeline used by this clone.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backend import QwenLocalClient
from .common import (
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


def _format_doc_map(doc_map: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for entry in doc_map:
        entities = ", ".join(entry.get("key_entities") or [])
        summary = compact_text(entry.get("summary", ""), limit=320)
        lines.append(
            f"- {entry['anchor_id']} | {entry.get('anchor_title','')}"
            + (f"\n    key_entities: {entities}" if entities else "")
            + (f"\n    summary: {summary}" if summary else "")
        )
    return "\n".join(lines) if lines else "(empty)"


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


def _normalized_contains(text: str, span: str) -> bool:
    normalized_text = normalize_ws(text)
    normalized_span = normalize_ws(span)
    return bool(normalized_span) and normalized_span in normalized_text


class ExtractAgent:
    """Extracts atomic facts from a single opened anchor."""

    def __init__(self, llm: QwenLocalClient, prompt_dir: Path) -> None:
        self.llm = llm
        self.system_prompt = (prompt_dir / "extract_system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (prompt_dir / "extract_user.txt").read_text(encoding="utf-8").strip()

    def run(
        self,
        *,
        query: str,
        instruction: str,
        doc_title: str,
        anchor: Dict[str, Any],
        need: str,
        evidence_so_far: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        user_prompt = self.user_template.format(
            query=query or "(empty)",
            instruction=instruction or "(none)",
            doc_title=doc_title,
            anchor_id=anchor["anchor_id"],
            anchor_title=anchor.get("anchor_title", ""),
            need=need or "(none)",
            evidence_so_far=_format_evidence(evidence_so_far),
            anchor_text=anchor.get("text", ""),
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=1200,
            metadata={"module": "extract_agent", "anchor_id": anchor["anchor_id"]},
        )
        think_text = normalize_ws(extract_tag_content(raw_text, "think") or "")
        info_text = extract_tag_content(raw_text, "info") or ""
        payload = extract_json_payload(info_text) or {}
        raw_items = payload.get("items", []) if isinstance(payload, dict) else []
        items: List[Dict[str, Any]] = []
        seen_keys = set()
        anchor_text = str(anchor.get("text", ""))
        anchor_summary = normalize_ws(anchor.get("summary", "")).casefold()
        if isinstance(raw_items, list):
            for it in raw_items:
                if not isinstance(it, dict):
                    continue
                fact = normalize_ws(it.get("fact", ""))
                evidence_span = normalize_ws(it.get("evidence_span", ""))
                if not fact or not evidence_span:
                    continue
                if not _normalized_contains(anchor_text, evidence_span):
                    continue
                if anchor_summary and fact.casefold() == anchor_summary:
                    continue
                key = (fact, evidence_span)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                items.append(
                    {
                        "fact": fact,
                        "entities": [normalize_ws(e) for e in (it.get("entities") or []) if normalize_ws(e)],
                        "evidence_span": evidence_span,
                        "relevance": normalize_ws(it.get("relevance", "")),
                        "source_doc_id": anchor["doc_id"],
                        "source_anchor_id": anchor["anchor_id"],
                    }
                )
        return {
            "think": think_text,
            "items": items,
            "raw_text": raw_text,
        }


class SearchAgent:
    def __init__(
        self,
        llm: QwenLocalClient,
        extract_agent: ExtractAgent,
        prompt_dir: Optional[Path] = None,
        max_rounds: int = 5,
        max_catalog_anchors: int = 12,
        local_window_size: int = 4,
    ) -> None:
        self.llm = llm
        self.extract_agent = extract_agent
        self.max_rounds = max_rounds
        self.max_catalog_anchors = max_catalog_anchors
        self.local_window_size = local_window_size
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parent / "prompts"
        self.system_prompt = (self.prompt_dir / "search_system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (self.prompt_dir / "search_user.txt").read_text(encoding="utf-8").strip()

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

    def initialize_state(self, *, query: str, instruction: str, doc_payload: Dict[str, Any]) -> SearchState:
        current_query = normalize_ws(query)
        must_find = self._initial_must_find(query=query, instruction=instruction)
        state = SearchState(
            current_query=current_query,
            must_find=must_find,
            global_frontier_anchor_ids=[anchor["anchor_id"] for anchor in doc_payload.get("anchors", [])],
        )
        state.missing_slots = self._summarize_missing_slots(query=query, instruction=instruction, state=state)
        return state

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

    def _anchor_neighbors(self, doc_payload: Dict[str, Any], anchor_id: str) -> List[str]:
        anchors = doc_payload.get("anchors", [])
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
                    normalize_ws(item.get("fact", ""))
                    + " "
                    + " ".join(item.get("entities") or [])
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
        doc_payload: Dict[str, Any],
        remaining_anchor_ids: List[str],
        state: SearchState,
    ) -> List[Tuple[float, Dict[str, Any], List[str]]]:
        allowed = set(remaining_anchor_ids)
        scored: List[Tuple[float, Dict[str, Any], List[str]]] = []
        for anchor in doc_payload.get("anchors", []):
            if anchor["anchor_id"] not in allowed:
                continue
            score, reasons = self._score_anchor(
                query=query,
                instruction=instruction,
                doc_title=doc_payload["doc_title"],
                anchor=anchor,
                state=state,
            )
            scored.append((score, anchor, reasons))
        scored.sort(key=lambda item: (item[0], -int(item[1].get("order", 0))), reverse=True)
        return scored

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

    def _plan_step(
        self,
        *,
        query: str,
        instruction: str,
        doc_payload: Dict[str, Any],
        state: SearchState,
        prior_trace: str,
    ) -> Dict[str, Any]:
        remaining_anchor_ids = [aid for aid in state.global_frontier_anchor_ids if aid not in state.seen_anchor_ids]
        scored = self._score_all_anchors(
            query=query,
            instruction=instruction,
            doc_payload=doc_payload,
            remaining_anchor_ids=remaining_anchor_ids,
            state=state,
        )
        shortlist = scored[: self.max_catalog_anchors]
        candidate_anchor_ids = self._candidate_anchor_ids(shortlist=shortlist, state=state)
        default_next_anchor_id = shortlist[0][1]["anchor_id"] if shortlist else ""
        action_type = "scan_neighbors" if state.local_frontier_anchor_ids else "inspect_anchor"
        next_query_if_fail = self._next_query_if_fail(query=query, instruction=instruction, state=state)

        result = {
            "think": (
                f"I should inspect one anchor from {doc_payload['doc_title']} that best matches the remaining need: "
                f"{', '.join(state.missing_slots or state.must_find) or 'task-relevant evidence'}."
            ),
            "action": "open" if default_next_anchor_id else "stop",
            "anchor_id": default_next_anchor_id,
            "need": "; ".join(state.missing_slots[:2] or state.must_find[:2]) or "task-relevant evidence",
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
        }

        if not shortlist:
            result["action"] = "stop"
            result["reason"] = "no plausible anchors remain"
            return result

        shortlist_doc_map = []
        for _, anchor, _ in shortlist[: min(len(shortlist), self.max_catalog_anchors)]:
            shortlist_doc_map.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "anchor_title": anchor.get("anchor_title", ""),
                    "summary": anchor.get("summary", ""),
                    "key_entities": anchor.get("key_entities", []),
                }
            )

        user_prompt = self.user_template.format(
            query=state.current_query or query or "(empty)",
            instruction=(instruction or "(none)")
            + "\n\nCurrent search state:\n"
            + self._state_summary(state)
            + "\n\nMissing slots:\n"
            + json.dumps(state.missing_slots, ensure_ascii=False),
            doc_title=doc_payload["doc_title"],
            doc_map=_format_doc_map(shortlist_doc_map),
            opened_anchors=", ".join(state.seen_anchor_ids) if state.seen_anchor_ids else "(none)",
            evidence_so_far=_format_evidence(state.known_items),
            round_index=state.round_index,
            max_rounds=self.max_rounds,
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=700,
            metadata={"module": "search_agent", "doc_title": doc_payload["doc_title"], "round": state.round_index},
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
            need = normalize_ws(str(payload.get("need", "")))
            reason = normalize_ws(str(payload.get("reason", "")))
            valid_ids = {anchor["anchor_id"] for _, anchor, _ in shortlist}
            if action in {"open", "stop"}:
                result["action"] = action
            if anchor_id in valid_ids:
                result["anchor_id"] = anchor_id
            if need:
                result["need"] = need
            if reason:
                result["reason"] = reason
        return result

    def _update_state_after_round(
        self,
        *,
        query: str,
        instruction: str,
        doc_payload: Dict[str, Any],
        state: SearchState,
        selected_anchor_id: str,
        extraction_items: List[Dict[str, Any]],
        next_query_if_fail: str,
        remaining_anchor_ids: List[str],
    ) -> SearchState:
        if selected_anchor_id and selected_anchor_id not in state.seen_anchor_ids:
            state.seen_anchor_ids.append(selected_anchor_id)
        state.last_anchor_id = selected_anchor_id
        state.round_index += 1
        state.global_frontier_anchor_ids = list(remaining_anchor_ids)
        state.local_frontier_anchor_ids = [
            anchor_id
            for anchor_id in self._anchor_neighbors(doc_payload, selected_anchor_id)
            if anchor_id in remaining_anchor_ids and anchor_id not in state.seen_anchor_ids
        ]
        state.known_items = list(extraction_items)
        previous_missing = list(state.missing_slots)
        state.missing_slots = self._summarize_missing_slots(query=query, instruction=instruction, state=state)
        if extraction_items:
            state.last_action = "scan_neighbors" if state.local_frontier_anchor_ids else "jump_to_section"
        elif next_query_if_fail and next_query_if_fail != state.current_query:
            state.failed_queries.append(state.current_query)
            state.current_query = next_query_if_fail
            state.last_action = "rewrite_query"
        else:
            state.last_action = "jump_to_section"
        if previous_missing == state.missing_slots and not extraction_items and next_query_if_fail:
            state.current_query = next_query_if_fail
        return state

    def run(
        self,
        *,
        query: str,
        instruction: str,
        doc_payload: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / f"{doc_payload['doc_id']}_search.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        anchors_by_id: Dict[str, Dict[str, Any]] = {a["anchor_id"]: a for a in doc_payload.get("anchors", [])}
        state = self.initialize_state(query=query, instruction=instruction, doc_payload=doc_payload)
        opened_ids: List[str] = []
        evidence: List[Dict[str, Any]] = []
        trace_lines: List[str] = []
        rounds: List[Dict[str, Any]] = []
        final_decision: Dict[str, Any] = {}

        for round_index in range(1, self.max_rounds + 1):
            remaining_ids = [aid for aid in anchors_by_id.keys() if aid not in opened_ids]
            state.global_frontier_anchor_ids = list(remaining_ids)
            state.round_index = round_index
            if not remaining_ids:
                break

            decision = self._plan_step(
                query=query,
                instruction=instruction,
                doc_payload=doc_payload,
                state=state,
                prior_trace="\n".join(trace_lines),
            )
            final_decision = decision

            action = decision["action"]
            anchor_id = decision.get("anchor_id", "")
            if action == "stop" and evidence:
                state.last_action = "stop_if_enough"
                trace_lines.append(f"<think>{decision['think']}</think>")
                trace_lines.append(
                    f"<search>{json.dumps({'action': 'stop', 'reason': decision.get('reason', '')}, ensure_ascii=False)}</search>"
                )
                rounds.append(
                    {
                        "round_index": round_index,
                        "plan": decision,
                        "anchor_id": "",
                        "info": None,
                        "extract": None,
                        "search_state": state.to_dict(),
                    }
                )
                break

            if anchor_id not in anchors_by_id or anchor_id in opened_ids:
                anchor_id = remaining_ids[0]
                decision["anchor_id"] = anchor_id
                decision["need"] = decision.get("need") or "fallback: inspect next unopened anchor"
                decision["reason"] = decision.get("reason") or "fallback to first unopened anchor"

            anchor = anchors_by_id[anchor_id]
            opened_ids.append(anchor_id)

            info_payload = {
                "anchor_id": anchor_id,
                "anchor_title": anchor.get("anchor_title", ""),
                "text": anchor.get("text", ""),
            }
            trace_lines.append(f"<think>{decision['think']}</think>")
            trace_lines.append(
                "<search>"
                + json.dumps(
                    {
                        "action": "open",
                        "anchor_id": anchor_id,
                        "need": decision.get("need", ""),
                        "action_type": decision.get("action_type", "inspect_anchor"),
                        "next_query_if_fail": decision.get("next_query_if_fail", state.current_query),
                    },
                    ensure_ascii=False,
                )
                + "</search>"
            )
            trace_lines.append(
                f"<info>{json.dumps({'anchor_id': anchor_id, 'anchor_title': info_payload['anchor_title']}, ensure_ascii=False)}</info>"
            )

            extraction = self.extract_agent.run(
                query=state.current_query or query,
                instruction=instruction,
                doc_title=doc_payload["doc_title"],
                anchor=anchor,
                need=decision.get("need", ""),
                evidence_so_far=evidence,
            )
            evidence.extend(extraction["items"])
            trace_lines.append(f"<think>{extraction['think']}</think>")
            trace_lines.append("<extracted>" + json.dumps({"items": extraction["items"]}, ensure_ascii=False) + "</extracted>")

            remaining_ids = [aid for aid in anchors_by_id.keys() if aid not in opened_ids]
            state = self._update_state_after_round(
                query=query,
                instruction=instruction,
                doc_payload=doc_payload,
                state=state,
                selected_anchor_id=anchor_id,
                extraction_items=evidence,
                next_query_if_fail=decision.get("next_query_if_fail", state.current_query),
                remaining_anchor_ids=remaining_ids,
            )

            rounds.append(
                {
                    "round_index": round_index,
                    "plan": decision,
                    "anchor_id": anchor_id,
                    "info": {"anchor_title": info_payload["anchor_title"], "char_count": len(info_payload["text"])},
                    "extract": {"think": extraction["think"], "item_count": len(extraction["items"])},
                    "search_state": state.to_dict(),
                }
            )

        payload_out = {
            "doc_id": doc_payload["doc_id"],
            "doc_title": doc_payload["doc_title"],
            "opened_anchor_ids": opened_ids,
            "items": evidence,
            "rounds": rounds,
            "current_query": state.current_query,
            "must_find": state.must_find,
            "action_type": final_decision.get("action_type", state.last_action),
            "next_query_if_fail": final_decision.get("next_query_if_fail", state.current_query),
            "candidate_anchor_ids": final_decision.get("candidate_anchor_ids", []),
            "ranked_anchor_ids": final_decision.get("ranked_anchor_ids", []),
            "all_ranked_anchor_ids": final_decision.get("all_ranked_anchor_ids", []),
            "rule_hits": final_decision.get("rule_hits", []),
            "search_state": state.to_dict(),
            "tagged_trace": "\n".join(trace_lines),
        }
        write_json(cache_path, payload_out)
        return payload_out
