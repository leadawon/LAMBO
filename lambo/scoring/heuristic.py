"""Heuristic anchor scoring — domain priors, query overlap, section priors.

These functions rank anchors WITHOUT LLM calls, using pattern matching and
domain knowledge. Used by DocRefineAgent to pre-sort anchors before the
LLM decides which to open.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from ..common import (
    compact_text,
    normalize_ws,
    quoted_terms,
    task_mode_for_level,
    tokenize_query,
)


def anchor_type_prior(record_type: str, level: int, anchor_type: str) -> int:
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


def section_prior(record_type: str, level: int, section_path: str) -> int:
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


def task_prior(level: int, summary: str, section_path: str) -> int:
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


def score_anchor(
    *,
    record: Dict[str, Any],
    doc_title: str,
    current_query: str,
    anchor: Dict[str, Any],
) -> Tuple[float, List[str]]:
    record_type = str(record.get("type", "")).strip().lower()
    level = int(record.get("level", 0) or 0)
    summary = str(anchor.get("summary", ""))
    section_path_val = str(anchor.get("section_path", ""))
    anchor_type_val = str(anchor.get("anchor_type", ""))
    combined = f"{doc_title} {section_path_val} {anchor_type_val} {summary}"
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

    section_score = section_prior(record_type, level, section_path_val)
    if section_score:
        score += section_score
        reasons.append(f"section_prior={section_score}")

    anchor_type_score_val = anchor_type_prior(record_type, level, anchor_type_val)
    if anchor_type_score_val:
        score += anchor_type_score_val
        reasons.append(f"anchor_type_prior={anchor_type_score_val}")

    task_score = task_prior(level, summary, section_path_val)
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


def score_all_anchors(
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
        s, reasons = score_anchor(
            record=record,
            doc_title=doc_payload["doc_title"],
            current_query=current_query,
            anchor=anchor,
        )
        scored.append((s, anchor, reasons))
    scored.sort(key=lambda item: (item[0], -int(item[1].get("order", 0))), reverse=True)
    return scored
