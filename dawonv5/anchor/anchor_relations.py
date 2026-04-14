from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .common import compact_text, normalize_ws, quoted_terms, tokenize_query


ANSWER_RELATION_TYPES = {"supports", "disambiguates", "prerequisite_of"}
CONFLICT_RELATION_TYPES = {"conflicts_with"}
SUPPORTIVE_RELATION_TYPES = ANSWER_RELATION_TYPES | {"same_section", "nearby_window", "likely_same_entity"}


def _add_unique(values: List[str], value: str) -> None:
    normalized = normalize_ws(value)
    if normalized and normalized not in values:
        values.append(normalized)


def _text(anchor: Dict[str, Any], doc_title: str = "") -> str:
    return " ".join(
        normalize_ws(part)
        for part in (
            doc_title,
            anchor.get("doc_title", ""),
            anchor.get("section_path", ""),
            anchor.get("anchor_type", ""),
            anchor.get("anchor_title", ""),
            anchor.get("summary", ""),
            compact_text(anchor.get("text", ""), limit=1600),
        )
        if normalize_ws(part)
    )


def _title_terms(title: str) -> List[str]:
    clean = re.sub(
        r"(《|》|20\d{2}年|第一季度|一季度|年度|季度|报告|股份有限公司|有限公司|公司|证券简称[:：]?)",
        " ",
        str(title or ""),
    )
    terms: List[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.:-]{2,}|[\u4e00-\u9fffA-Za-z0-9·]{2,36}", clean):
        normalized = normalize_ws(token)
        if normalized and normalized.casefold() not in {"doc", "paper", "report", "title", "公司", "报告"}:
            _add_unique(terms, normalized)
    return terms[:10]


def _metric_terms(text: str) -> List[str]:
    terms: List[str] = []
    metric_pattern = re.compile(
        r"(营业收入|净利润|扣除非经常性损益|现金流量净额|投资现金流量净额|经营活动产生的现金流量净额|"
        r"应收票据|应收账款|总资产|负债合计|所有者权益|每股收益|毛利率|收入|利润|现金|资产|负债|权益|收益|费用|净额|金额|增长|同比|余额|"
        r"revenue|profit|cash|asset|liabilit)",
        re.IGNORECASE,
    )
    for match in metric_pattern.findall(str(text or "")):
        _add_unique(terms, match)
    for term in quoted_terms(text):
        if re.search(metric_pattern, term):
            _add_unique(terms, term)
    return terms[:12]


def _numeric_hints(text: str) -> List[str]:
    hints: List[str] = []
    for match in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|元|美元|股)?", str(text or "")):
        _add_unique(hints, match)
        if len(hints) >= 8:
            break
    return hints


def _target_terms(record: Dict[str, Any], doc_title: str, text: str) -> List[str]:
    terms: List[str] = []
    task_text = f"{record.get('question', '')}\n{record.get('instruction', '')}\n{doc_title}\n{text}"
    for term in quoted_terms(task_text):
        if len(term) > 2 and term.casefold() not in {"reference", "references", "citation", "citations"}:
            _add_unique(terms, term)
    for term in _title_terms(doc_title):
        _add_unique(terms, term)
    return terms[:12]


def _anchor_signature(anchor: Dict[str, Any]) -> str:
    return " ".join(
        str(item)
        for item in (
            anchor.get("anchor_id", ""),
            anchor.get("section_path", ""),
            anchor.get("anchor_type", ""),
            anchor.get("summary", ""),
        )
    )


def _token_overlap(left: Iterable[str], right: Iterable[str]) -> List[str]:
    right_set = {normalize_ws(item).casefold() for item in right if normalize_ws(item)}
    hits: List[str] = []
    for item in left:
        normalized = normalize_ws(item)
        if normalized and normalized.casefold() in right_set and normalized not in hits:
            hits.append(normalized)
    return hits


def annotate_anchor(record: Dict[str, Any], doc_title: str, anchor: Dict[str, Any]) -> Dict[str, Any]:
    record_type = str(record.get("type", "")).strip().lower()
    level = int(record.get("level", 0) or 0)
    text = _text(anchor, doc_title)
    lowered = text.casefold()
    section = normalize_ws(anchor.get("section_path", ""))
    anchor_type = normalize_ws(anchor.get("anchor_type", ""))
    roles: List[str] = []
    entities: List[str] = []
    targets: List[str] = []
    value_hints = _numeric_hints(text)
    provenance_signals: List[str] = []

    for term in _title_terms(doc_title):
        if term.casefold() in lowered or record_type in {"financial", "legal", "paper"}:
            _add_unique(entities, term)
    for term in _target_terms(record, doc_title, text):
        if term.casefold() in lowered:
            _add_unique(targets, term)

    metric_terms = _metric_terms(text)
    for term in metric_terms:
        _add_unique(targets, term)

    if anchor_type == "table_region" or "|" in str(anchor.get("text", "")):
        _add_unique(roles, "table_value")
        provenance_signals.append("table_region")
    if value_hints:
        _add_unique(roles, "value_evidence")
        provenance_signals.append("numeric_value")
    if metric_terms:
        _add_unique(roles, "metric_binding")
        provenance_signals.append("metric_term")

    if record_type == "financial":
        if entities:
            _add_unique(roles, "entity_binding")
        if anchor_type != "table_region" and not value_hints:
            _add_unique(roles, "background_context")

    elif record_type == "paper":
        if re.search(r"(reference|bibliography|参考文献|\[[0-9]+\]|et al\.)", lowered):
            _add_unique(roles, "citation_direction")
            provenance_signals.append("citation_marker")
        if any(term.casefold() in normalize_ws(doc_title).casefold() for term in targets):
            _add_unique(roles, "entity_binding")
        if not roles:
            _add_unique(roles, "background_context")

    elif record_type == "legal":
        if re.search(r"(案由|原告|被告|法院|裁定书|判决书|行政|刑事|民事|纠纷)", lowered):
            _add_unique(roles, "case_identity")
            provenance_signals.append("case_identity_cue")
        if re.search(r"(本院认为|判决如下|裁判|判决结果|撤销|驳回|赔偿|罪|裁定如下)", lowered):
            _add_unique(roles, "decision_evidence")
            provenance_signals.append("decision_cue")
        if not roles:
            _add_unique(roles, "background_context")

    if level in {3, 4} and re.search(r"(分类|category|group|chain|trend|变化|结果|mapping)", lowered):
        _add_unique(roles, "background_context")

    if not roles:
        _add_unique(roles, "background_context")

    strength = "strong" if {"value_evidence", "table_value", "decision_evidence", "citation_direction"} & set(roles) else "medium"
    if roles == ["background_context"]:
        strength = "weak"

    return {
        "anchor_role_candidates": roles,
        "anchor_entities": entities[:12],
        "anchor_targets": targets[:16],
        "anchor_value_hints": value_hints,
        "provenance_hints": {
            "strength": strength,
            "signals": provenance_signals[:8],
            "record_type": record_type,
        },
    }


def _relation(source: Dict[str, Any], target: Dict[str, Any], relation_type: str, weight: float, reason: str) -> Dict[str, Any]:
    return {
        "relation_type": relation_type,
        "target_anchor_id": target["anchor_id"],
        "weight": round(float(weight), 3),
        "reason": reason,
    }


def _add_relation(relations: Dict[str, List[Dict[str, Any]]], source: Dict[str, Any], target: Dict[str, Any], relation_type: str, weight: float, reason: str) -> None:
    if source["anchor_id"] == target["anchor_id"]:
        return
    candidate = _relation(source, target, relation_type, weight, reason)
    existing = relations[source["anchor_id"]]
    key = (candidate["relation_type"], candidate["target_anchor_id"])
    if any((item["relation_type"], item["target_anchor_id"]) == key for item in existing):
        return
    existing.append(candidate)


def _has_role(anchor: Dict[str, Any], role: str) -> bool:
    return role in set(anchor.get("anchor_role_candidates", []) or [])


def _shared_entities(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    return _token_overlap(left.get("anchor_entities", []) or [], right.get("anchor_entities", []) or [])


def _shared_targets(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    return _token_overlap(left.get("anchor_targets", []) or [], right.get("anchor_targets", []) or [])


def build_anchor_relations(record: Dict[str, Any], anchors: List[Dict[str, Any]]) -> Dict[str, Any]:
    record_type = str(record.get("type", "")).strip().lower()
    relations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    ordered = sorted(anchors, key=lambda item: int(item.get("order", 0)))

    for idx, anchor in enumerate(ordered):
        if idx > 0:
            _add_relation(relations, anchor, ordered[idx - 1], "prev_next", 1.4, "previous anchor")
        if idx + 1 < len(ordered):
            _add_relation(relations, anchor, ordered[idx + 1], "prev_next", 1.4, "next anchor")
        for other_idx in range(max(0, idx - 2), min(len(ordered), idx + 3)):
            if other_idx == idx:
                continue
            other = ordered[other_idx]
            if normalize_ws(anchor.get("section_path", "")) == normalize_ws(other.get("section_path", "")):
                _add_relation(relations, anchor, other, "same_section", 2.0, "same section_path")
            _add_relation(relations, anchor, other, "nearby_window", 1.0, "within local read window")

    for idx, left in enumerate(ordered):
        for right in ordered[idx + 1 :]:
            shared_entities = _shared_entities(left, right)
            shared_targets = _shared_targets(left, right)
            if shared_entities:
                _add_relation(relations, left, right, "likely_same_entity", 2.2, f"shared entity {shared_entities[:2]}")
                _add_relation(relations, right, left, "likely_same_entity", 2.2, f"shared entity {shared_entities[:2]}")

            if record_type == "financial" and shared_entities and shared_targets:
                reason = f"shared company/metric {shared_entities[:1]} {shared_targets[:2]}"
                _add_relation(relations, left, right, "same_company_metric", 3.0, reason)
                _add_relation(relations, right, left, "same_company_metric", 3.0, reason)

            if record_type == "paper" and (shared_targets or shared_entities):
                reason = f"shared paper target {(shared_targets or shared_entities)[:2]}"
                _add_relation(relations, left, right, "same_target_paper", 2.8, reason)
                _add_relation(relations, right, left, "same_target_paper", 2.8, reason)

            if record_type == "legal" and shared_entities:
                reason = f"shared case cue {shared_entities[:2]}"
                _add_relation(relations, left, right, "same_case", 2.8, reason)
                _add_relation(relations, right, left, "same_case", 2.8, reason)

            left_roles = set(left.get("anchor_role_candidates", []) or [])
            right_roles = set(right.get("anchor_role_candidates", []) or [])
            if {"value_evidence", "table_value", "decision_evidence", "citation_direction"} & left_roles and {
                "entity_binding",
                "metric_binding",
                "case_identity",
            } & right_roles:
                _add_relation(relations, left, right, "supports", 3.4, "evidence supports binding/identity anchor")
                _add_relation(relations, right, left, "disambiguates", 2.8, "binding anchor disambiguates evidence")
            if {"value_evidence", "table_value", "decision_evidence", "citation_direction"} & right_roles and {
                "entity_binding",
                "metric_binding",
                "case_identity",
            } & left_roles:
                _add_relation(relations, right, left, "supports", 3.4, "evidence supports binding/identity anchor")
                _add_relation(relations, left, right, "disambiguates", 2.8, "binding anchor disambiguates evidence")

            if "background_context" in left_roles and right_roles - {"background_context"}:
                _add_relation(relations, left, right, "prerequisite_of", 1.8, "background precedes answer evidence")
            if "background_context" in right_roles and left_roles - {"background_context"}:
                _add_relation(relations, right, left, "prerequisite_of", 1.8, "background precedes answer evidence")

            left_values = set(left.get("anchor_value_hints", []) or [])
            right_values = set(right.get("anchor_value_hints", []) or [])
            if shared_targets and left_values and right_values and not (left_values & right_values):
                _add_relation(relations, left, right, "conflicts_with", 2.5, f"different values for shared target {shared_targets[:2]}")
                _add_relation(relations, right, left, "conflicts_with", 2.5, f"different values for shared target {shared_targets[:2]}")

    relation_type_counts: Counter[str] = Counter()
    for anchor in anchors:
        anchor_id = anchor["anchor_id"]
        anchor_relations = sorted(
            relations.get(anchor_id, []),
            key=lambda item: (float(item.get("weight", 0.0)), item.get("relation_type", ""), item.get("target_anchor_id", "")),
            reverse=True,
        )[:18]
        for relation in anchor_relations:
            relation_type_counts[relation["relation_type"]] += 1
        anchor["anchor_relations"] = anchor_relations
        anchor["relation_summary"] = {
            "relation_count": len(anchor_relations),
            "relation_type_counts": dict(Counter(item["relation_type"] for item in anchor_relations)),
            "top_related_anchor_ids": [item["target_anchor_id"] for item in anchor_relations[:6]],
        }

    return {
        "relation_count": sum(relation_type_counts.values()),
        "relation_type_counts": dict(sorted(relation_type_counts.items())),
    }


def enrich_anchor_graph(record: Dict[str, Any], doc_title: str, anchors: List[Dict[str, Any]]) -> Dict[str, Any]:
    for anchor in anchors:
        anchor.update(annotate_anchor(record, doc_title, anchor))
    return build_anchor_relations(record, anchors)


def relation_context(anchor: Dict[str, Any], *, selected_anchor_ids: Iterable[str] = (), max_items: int = 6) -> List[Dict[str, Any]]:
    selected = {normalize_ws(item) for item in selected_anchor_ids if normalize_ws(item)}
    rows: List[Dict[str, Any]] = []
    for relation in anchor.get("anchor_relations", []) or []:
        if selected and relation.get("target_anchor_id") not in selected:
            continue
        rows.append(
            {
                "relation_type": relation.get("relation_type", ""),
                "target_anchor_id": relation.get("target_anchor_id", ""),
                "weight": relation.get("weight", 0.0),
                "reason": relation.get("reason", ""),
            }
        )
        if len(rows) >= max_items:
            break
    return rows


def relation_bias_for_anchor(anchor: Dict[str, Any], coverage_hints: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    roles = list(anchor.get("anchor_role_candidates", []) or [])
    role_set = set(roles)
    selected = {normalize_ws(item) for item in coverage_hints.get("selected_anchor_ids", []) or [] if normalize_ws(item)}
    relation_anchor_ids = {
        normalize_ws(item)
        for item in coverage_hints.get("selected_relation_anchor_ids", []) or []
        if normalize_ws(item)
    }
    missing_text = " ".join(
        normalize_ws(item)
        for key in ("missing_answer_keys", "missing_must_find", "must_find", "low_confidence_keys")
        for item in (coverage_hints.get(key, []) or [])
    ).casefold()

    score = 0.0
    relation_hits: List[str] = []
    role_hits: List[str] = []

    if "table" in missing_text or "numeric" in missing_text or "数" in missing_text or "表" in missing_text:
        for role in ("table_value", "value_evidence", "metric_binding"):
            if role in role_set:
                score += 2.5
                role_hits.append(role)
    if "citation" in missing_text or "reference" in missing_text or "引用" in missing_text or "参考" in missing_text:
        if "citation_direction" in role_set:
            score += 5.0
            role_hits.append("citation_direction")
    if "case" in missing_text or "verdict" in missing_text or "判决" in missing_text or "案由" in missing_text:
        for role in ("case_identity", "decision_evidence"):
            if role in role_set:
                score += 3.0
                role_hits.append(role)
    if coverage_hints.get("needs_non_doc_placeholder") and {"entity_binding", "case_identity", "metric_binding"} & role_set:
        score += 2.2
        role_hits.extend(sorted({"entity_binding", "case_identity", "metric_binding"} & role_set))

    anchor_id = normalize_ws(anchor.get("anchor_id", ""))
    if anchor_id in relation_anchor_ids:
        score += 3.0
        relation_hits.append("candidate_in_selected_relation_frontier")

    for relation in anchor.get("anchor_relations", []) or []:
        relation_type = normalize_ws(relation.get("relation_type", ""))
        target_id = normalize_ws(relation.get("target_anchor_id", ""))
        weight = float(relation.get("weight", 0.0) or 0.0)
        if target_id in selected:
            if relation_type in ANSWER_RELATION_TYPES:
                delta = 2.5 + min(weight, 3.5) * 0.4
                score += delta
                relation_hits.append(f"{relation_type}_selected={target_id}")
            elif relation_type in {"same_section", "nearby_window", "same_company_metric", "same_case", "same_target_paper", "likely_same_entity"}:
                delta = 1.0 + min(weight, 3.0) * 0.25
                score += delta
                relation_hits.append(f"{relation_type}_selected={target_id}")
            elif relation_type in CONFLICT_RELATION_TYPES:
                delta = 3.0 + min(weight, 3.0) * 0.5
                score -= delta
                relation_hits.append(f"conflicts_with_selected={target_id}")

    return round(score, 3), relation_hits[:10], sorted(set(role_hits))[:10]


def provenance_strength_from_anchor(anchor: Dict[str, Any]) -> Tuple[str, bool]:
    strength = "weak"
    hints = anchor.get("provenance_hints", {})
    if isinstance(hints, dict):
        strength = normalize_ws(hints.get("strength", "")) or strength
    relation_types = {normalize_ws(item.get("relation_type", "")) for item in anchor.get("anchor_relations", []) or []}
    if relation_types & ANSWER_RELATION_TYPES and strength == "medium":
        strength = "strong"
    roles = set(anchor.get("anchor_role_candidates", []) or [])
    disambiguation_needed = not bool(roles & {"value_evidence", "table_value", "decision_evidence", "citation_direction"})
    if relation_types & {"disambiguates", "supports"}:
        disambiguation_needed = False
    return strength, disambiguation_needed
