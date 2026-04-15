from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backend import QwenLocalClient
from .common import (
    compact_text,
    current_query_from_record,
    extract_json_payload,
    extract_tag_content,
    normalize_ws,
    quoted_terms,
    read_json,
    task_mode_for_level,
    tokenize_query,
    write_json,
)


class SearchR1:
    def __init__(
        self,
        llm: Optional[QwenLocalClient] = None,
        prompt_dir: Optional[Path] = None,
        use_llm_planning: bool = True,
        max_catalog_anchors: int = 12,
    ) -> None:
        self.llm = llm
        self.use_llm_planning = use_llm_planning and llm is not None
        self.max_catalog_anchors = max_catalog_anchors
        self.prompt_dir = prompt_dir or Path(__file__).resolve().parent / "prompts"
        self.system_prompt = (self.prompt_dir / "search_r1_system.txt").read_text(encoding="utf-8").strip()
        self.user_prompt = (self.prompt_dir / "search_r1_user.txt").read_text(encoding="utf-8").strip()

    def _heuristic_plan(self, record: Dict[str, Any], doc_payload: Dict[str, Any], current_query: str) -> Dict[str, Any]:
        task_mode = task_mode_for_level(int(record.get("level", 0) or 0))
        must_find = []
        if task_mode == "spotlight":
            must_find.append("direct answer evidence")
        elif task_mode == "comparison":
            must_find.append("candidate entities and comparable values")
        elif task_mode == "clustering":
            must_find.append("category labels and document-to-category assignments")
        elif task_mode == "chain":
            must_find.append("ordered relations or decision/result mappings")
        doc_type = str(record.get("type", "")).strip().lower()
        if doc_type == "paper":
            must_find.append("citation/reference evidence")
        elif doc_type == "financial":
            must_find.append("tables or numeric evidence")
        elif doc_type == "legal":
            must_find.append("case cause, disposition, or verdict evidence")
        think = (
            f"I should inspect one anchor from {doc_payload['doc_title']} that best matches the current need: "
            f"{', '.join(must_find) or 'task-relevant evidence'}."
        )
        return {
            "think": think,
            "current_query": normalize_ws(current_query),
            "must_find": must_find,
        }

    @staticmethod
    def _anchor_type_prior(record_type: str, level: int, anchor_type: str) -> int:
        record_type = record_type.lower()
        if record_type == "financial":
            if anchor_type == "table_region":
                return 12 if level in {1, 2, 3} else 8
            if anchor_type == "paragraph_region":
                return 4
        if record_type == "legal":
            if anchor_type in {"paragraph_region", "clause_region"}:
                return 8
            if anchor_type == "table_region":
                return -2
        if record_type == "paper":
            if anchor_type == "attribution_region":
                return 12
            if anchor_type == "paragraph_region":
                return 6
        return 0

    @staticmethod
    def _section_prior(record_type: str, level: int, section_path: str) -> int:
        lowered = section_path.casefold()
        score = 0
        if record_type == "paper":
            if re.search(r"(reference|bibliography|引用|参考文献)", lowered):
                score += 14
            if re.search(r"(abstract|introduction|related work|preliminar|method|experiment|conclusion)", lowered):
                score += 4
        elif record_type == "financial":
            if re.search(r"(主要财务数据|财务报表|季度财务报表|资产负债表|利润表|现金流量表|notes)", lowered):
                score += 10
        elif record_type == "legal":
            if re.search(r"(本院认为|经审理查明|裁判|判决结果|案由|行政|刑事|民事)", lowered):
                score += 10
        if level == 4 and re.search(r"(result|结果|判决)", lowered):
            score += 6
        return score

    @staticmethod
    def _task_prior(level: int, summary: str, section_path: str) -> int:
        text = f"{summary} {section_path}".casefold()
        score = 0
        if level == 1 and re.search(r"(answer|which|what|是多少|哪一个|案由)", text):
            score += 5
        elif level == 2 and re.search(r"(highest|lowest|排序|比较|顺序|最高|最低|largest|smallest)", text):
            score += 8
        elif level == 3 and re.search(r"(分类|cluster|category|group|reference|citation)", text):
            score += 8
        elif level == 4 and re.search(r"(chain|trend|结果|citation|growth|变化趋势|mapping)", text):
            score += 8
        return score

    def _score_anchor(
        self,
        *,
        record: Dict[str, Any],
        doc_title: str,
        current_query: str,
        anchor: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        record_type = str(record.get("type", "")).strip().lower()
        level = int(record.get("level", 0) or 0)
        summary = str(anchor.get("summary", ""))
        section_path = str(anchor.get("section_path", ""))
        anchor_type = str(anchor.get("anchor_type", ""))
        combined = f"{doc_title} {section_path} {anchor_type} {summary}"
        combined_lower = combined.casefold()

        query_tokens = tokenize_query(current_query)
        instruction_tokens = tokenize_query(str(record.get("instruction", "")))
        quote_tokens = quoted_terms(str(record.get("question", "")) + "\n" + str(record.get("instruction", "")))

        score = 0.0
        reasons: List[str] = []

        overlap = sum(1 for token in query_tokens if token in combined_lower)
        if overlap:
            score += overlap * 8
            reasons.append(f"query_overlap={overlap}")

        instruction_overlap = sum(1 for token in instruction_tokens[:12] if token in combined_lower)
        if instruction_overlap:
            score += min(instruction_overlap, 4) * 3
            reasons.append(f"instruction_overlap={instruction_overlap}")

        quote_hits = sum(1 for token in quote_tokens if token.casefold() in combined_lower)
        if quote_hits:
            score += quote_hits * 12
            reasons.append(f"quoted_hit={quote_hits}")

        if any(re.search(r"\d", token) for token in query_tokens) and re.search(r"\d", combined):
            score += 5
            reasons.append("numeric_match")

        section_score = self._section_prior(record_type, level, section_path)
        if section_score:
            score += section_score
            reasons.append(f"section_prior={section_score}")

        anchor_type_score = self._anchor_type_prior(record_type, level, anchor_type)
        if anchor_type_score:
            score += anchor_type_score
            reasons.append(f"anchor_type_prior={anchor_type_score}")

        task_score = self._task_prior(level, summary, section_path)
        if task_score:
            score += task_score
            reasons.append(f"task_prior={task_score}")

        if record_type == "paper" and re.search(r"(reference|citation|et al\.|\[[0-9]+\])", combined_lower):
            score += 10
            reasons.append("citation_cue")

        if record_type == "financial" and re.search(r"(元|%|revenue|profit|cash|资产|利润|收入|应收)", combined_lower):
            score += 6
            reasons.append("financial_cue")

        if record_type == "legal" and re.search(r"(案由|判决|裁定|撤销|驳回|罪|纠纷|行政)", combined_lower):
            score += 6
            reasons.append("legal_cue")

        position_bonus = max(0.0, 2.5 - math.log(anchor.get("order", 1) + 1))
        score += position_bonus
        reasons.append(f"position_bonus={position_bonus:.2f}")
        return score, reasons

    def _score_all_anchors(
        self,
        *,
        record: Dict[str, Any],
        doc_payload: Dict[str, Any],
        current_query: str,
        remaining_anchor_ids: Optional[List[str]] = None,
    ) -> List[Tuple[float, Dict[str, Any], List[str]]]:
        allowed = set(remaining_anchor_ids) if remaining_anchor_ids else None
        scored: List[Tuple[float, Dict[str, Any], List[str]]] = []
        for anchor in doc_payload.get("anchors", []):
            if allowed is not None and anchor["anchor_id"] not in allowed:
                continue
            score, reasons = self._score_anchor(
                record=record,
                doc_title=doc_payload["doc_title"],
                current_query=current_query,
                anchor=anchor,
            )
            scored.append((score, anchor, reasons))
        scored.sort(key=lambda item: (item[0], -int(item[1].get("order", 0))), reverse=True)
        return scored

    @staticmethod
    def _chunk_summary_text(scored_anchors: List[Tuple[float, Dict[str, Any], List[str]]]) -> str:
        blocks: List[str] = []
        for _, anchor, _ in scored_anchors:
            blocks.append(
                "\n".join(
                    [
                        anchor["anchor_id"],
                        normalize_ws(anchor.get("section_path", "")) or "(root)",
                        normalize_ws(anchor.get("summary", "")) or "(no summary)",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def choose_next_anchor(
        self,
        *,
        record: Dict[str, Any],
        doc_payload: Dict[str, Any],
        remaining_anchor_ids: Optional[List[str]] = None,
        prior_trace: str = "",
        known_items: Optional[List[Dict[str, Any]]] = None,
        current_query_override: Optional[str] = None,
        must_find_override: Optional[List[str]] = None,
        round_index: int = 1,
    ) -> Dict[str, Any]:
        base_query = current_query_override or current_query_from_record(
            str(record.get("question", "")).strip(),
            str(record.get("instruction", "")).strip(),
        )
        fallback = self._heuristic_plan(record, doc_payload, base_query)
        if must_find_override:
            fallback["must_find"] = [normalize_ws(item) for item in must_find_override if normalize_ws(item)]

        scored = self._score_all_anchors(
            record=record,
            doc_payload=doc_payload,
            current_query=fallback["current_query"],
            remaining_anchor_ids=remaining_anchor_ids,
        )
        shortlist = scored[: self.max_catalog_anchors]
        default_next_anchor_id = shortlist[0][1]["anchor_id"] if shortlist else ""
        known_items = known_items or []

        result = {
            "think": fallback["think"],
            "current_query": fallback["current_query"],
            "must_find": fallback["must_find"],
            "next_anchor_id": default_next_anchor_id,
            "candidate_anchor_ids": [anchor["anchor_id"] for _, anchor, _ in shortlist[:4]],
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
            "raw_plan_text": "",
        }

        if not self.use_llm_planning or not shortlist:
            return result

        prompt = self.user_prompt.format(
            record_type=record.get("type"),
            level=record.get("level"),
            task_mode=task_mode_for_level(int(record.get("level", 0) or 0)),
            round_index=round_index,
            question=str(record.get("question", "")).strip() or "(empty)",
            instruction=str(record.get("instruction", "")).strip(),
            doc_title=doc_payload["doc_title"],
            current_query=result["current_query"] or "(empty)",
            must_find_json=json.dumps(result["must_find"], ensure_ascii=False),
            previous_trace=prior_trace.strip() or "(empty)",
            known_items_json=json.dumps(known_items[-6:], ensure_ascii=False, indent=2),
            chunk_summary_text=self._chunk_summary_text(shortlist),
        )
        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            max_output_tokens=900,
            metadata={
                "module": "search_r1",
                "doc_title": doc_payload["doc_title"],
                "round_index": round_index,
            },
        )
        result["raw_plan_text"] = raw_text
        think_text = extract_tag_content(raw_text, "think")
        search_text = extract_tag_content(raw_text, "search")
        if think_text:
            result["think"] = normalize_ws(think_text)

        payload = extract_json_payload(search_text or "")
        if isinstance(payload, dict):
            current_query = normalize_ws(payload.get("current_query", "")) or result["current_query"]
            rescored = self._score_all_anchors(
                record=record,
                doc_payload=doc_payload,
                current_query=current_query,
                remaining_anchor_ids=remaining_anchor_ids,
            )
            shortlist = rescored[: self.max_catalog_anchors]
            result["current_query"] = current_query
            if isinstance(payload.get("must_find"), list):
                result["must_find"] = [normalize_ws(item) for item in payload["must_find"] if normalize_ws(item)]
            if isinstance(payload.get("candidate_anchor_ids"), list):
                requested_ids = [normalize_ws(item) for item in payload["candidate_anchor_ids"] if normalize_ws(item)]
            else:
                requested_ids = []
            next_anchor_id = normalize_ws(payload.get("next_anchor_id", ""))
            valid_ids = {anchor["anchor_id"] for _, anchor, _ in rescored}
            if next_anchor_id in valid_ids:
                result["next_anchor_id"] = next_anchor_id
            elif requested_ids:
                for anchor_id in requested_ids:
                    if anchor_id in valid_ids:
                        result["next_anchor_id"] = anchor_id
                        break
            elif shortlist:
                result["next_anchor_id"] = shortlist[0][1]["anchor_id"]
            result["candidate_anchor_ids"] = requested_ids[:4] or [anchor["anchor_id"] for _, anchor, _ in shortlist[:4]]
            result["ranked_anchor_ids"] = [anchor["anchor_id"] for _, anchor, _ in rescored[:4]]
            result["all_ranked_anchor_ids"] = [anchor["anchor_id"] for _, anchor, _ in rescored]
            result["rule_hits"] = [
                {
                    "anchor_id": anchor["anchor_id"],
                    "score": round(score, 3),
                    "reasons": reasons,
                    "summary": compact_text(anchor.get("summary", ""), limit=180),
                }
                for score, anchor, reasons in rescored[:4]
            ]
        elif search_text:
            result["current_query"] = normalize_ws(search_text)

        return result

    def run(
        self,
        *,
        record: Dict[str, Any],
        doc_payload: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / f"{doc_payload['doc_id']}_search.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        decision = self.choose_next_anchor(record=record, doc_payload=doc_payload)
        why = (
            f"Prepared iterative anchor search for {doc_payload['doc_title']}. "
            f"First coordinate candidate is {decision.get('next_anchor_id', '(none)')}."
        )
        search_payload = {
            "current_query": decision.get("current_query", ""),
            "must_find": decision.get("must_find", []),
            "candidate_anchor_ids": decision.get("candidate_anchor_ids", []),
            "next_anchor_id": decision.get("next_anchor_id", ""),
        }
        tagged_trace = (
            f"<think>{decision.get('think', '')}</think>\n"
            f"<search>{json.dumps(search_payload, ensure_ascii=False)}</search>"
        )
        payload = {
            "doc_id": doc_payload["doc_id"],
            "doc_title": doc_payload["doc_title"],
            "task_mode": task_mode_for_level(int(record.get("level", 0) or 0)),
            "think": decision.get("think", ""),
            "current_query": decision.get("current_query", ""),
            "must_find": decision.get("must_find", []),
            "next_anchor_id": decision.get("next_anchor_id", ""),
            "candidate_anchor_ids": decision.get("candidate_anchor_ids", []),
            "ranked_anchor_ids": decision.get("ranked_anchor_ids", []),
            "all_ranked_anchor_ids": decision.get("all_ranked_anchor_ids", []),
            "rule_hits": decision.get("rule_hits", []),
            "why_these_anchors": why,
            "raw_plan_text": decision.get("raw_plan_text", ""),
            "tagged_trace": tagged_trace,
        }
        write_json(cache_path, payload)
        return payload
