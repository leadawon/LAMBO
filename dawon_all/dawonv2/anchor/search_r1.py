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

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        normalized = normalize_ws(term).casefold()
        return bool(normalized and normalized in str(text or "").casefold())

    @staticmethod
    def _title_terms(title: str) -> List[str]:
        clean = re.sub(
            r"(《|》|2024年|2023年|2022年|第一季度|一季度|年度|季度|报告|股份有限公司|有限公司|公司|证券简称[:：]?)",
            " ",
            str(title or ""),
        )
        generic = {"doc", "paper", "report", "title", "年度", "季度", "报告", "公司", "股份", "有限"}
        terms: List[str] = []
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{1,}|[\u4e00-\u9fffA-Za-z0-9·]{2,24}", clean):
            term = normalize_ws(token)
            if not term or term.casefold() in generic:
                continue
            if term not in terms:
                terms.append(term)
        return terms[:8]

    @staticmethod
    def _important_terms(text: str) -> List[str]:
        stop_terms = {
            "answer",
            "question",
            "which",
            "what",
            "company",
            "document",
            "documents",
            "paper",
            "papers",
            "reference",
            "references",
            "citation",
            "citations",
            "请回答",
            "公司名称",
            "公司",
            "名称",
            "报告",
            "文档",
            "论文",
            "引用",
            "参考文献",
            "哪些",
            "哪一个",
            "是多少",
            "根据",
            "仅根据",
        }
        terms: List[str] = []
        for term in quoted_terms(text):
            normalized = normalize_ws(term)
            if normalized and normalized.casefold() not in stop_terms and normalized not in terms:
                terms.append(normalized)
        for token in tokenize_query(text):
            normalized = normalize_ws(token)
            if len(normalized) <= 1 or len(normalized) > 36:
                continue
            if normalized.casefold() in stop_terms:
                continue
            if normalized not in terms:
                terms.append(normalized)
        return terms[:24]

    @staticmethod
    def _query_entity_terms(text: str) -> List[str]:
        finance_words = r"(收入|利润|现金|资产|负债|权益|收益|费用|净额|金额|增长|同比|余额|每股|毛利|应收|应付|投资)"
        terms: List[str] = []
        patterns = [
            r"([A-Za-z0-9\u4e00-\u9fff·]{2,18})的",
            r"([A-Za-z0-9\u4e00-\u9fff·]{2,24})(?:股份有限公司|有限公司|公司)",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, str(text or "")):
                term = re.sub(r"^(请回答|回答|请问|请)", "", normalize_ws(match))
                if not term or "为" in term or re.search(finance_words, term):
                    continue
                if term not in terms:
                    terms.append(term)
        return terms[:8]

    @staticmethod
    def _paper_direction(text: str) -> str:
        lowered = str(text or "").casefold()
        citation_hit = re.search(r"(citation|cited by|被引用|引用它|引用该|cites the target)", lowered)
        reference_hit = re.search(r"(reference|references|bibliography|参考文献|所引用|cites)", lowered)
        if citation_hit and not reference_hit:
            return "citation"
        if reference_hit and not citation_hit:
            return "reference"
        if citation_hit and reference_hit:
            return "both"
        return ""

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        left_terms = set(tokenize_query(left))
        right_terms = set(tokenize_query(right))
        if not left_terms or not right_terms:
            return 0.0
        return len(left_terms & right_terms) / max(1, len(left_terms | right_terms))

    @staticmethod
    def _known_item_hints(known_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        answer_keys: List[str] = []
        value_signatures: List[str] = []
        low_confidence_keys: List[str] = []
        for item in known_items:
            key = normalize_ws(item.get("answer_key", ""))
            if key and key not in answer_keys:
                answer_keys.append(key)
            value = item.get("normalized_value", item.get("value", ""))
            value_text = normalize_ws(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
            if value_text and value_text not in value_signatures:
                value_signatures.append(value_text)
            try:
                confidence = float(item.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            if confidence < 0.45 and key and key not in low_confidence_keys:
                low_confidence_keys.append(key)
        return {
            "known_answer_keys": answer_keys,
            "known_value_signatures": value_signatures[:12],
            "low_confidence_keys": low_confidence_keys,
        }

    def _task_utility_score(
        self,
        *,
        record: Dict[str, Any],
        doc_title: str,
        current_query: str,
        anchor: Dict[str, Any],
        combined_text: str,
    ) -> Tuple[float, List[str]]:
        record_type = str(record.get("type", "")).strip().lower()
        query_text = "\n".join(
            [
                current_query,
                str(record.get("question", "")),
                str(record.get("instruction", "")),
            ]
        )
        text_lower = combined_text.casefold()
        utility = 0.0
        hits: List[str] = []

        if record_type == "financial":
            metric_terms = [
                term
                for term in self._important_terms(query_text)
                if re.search(r"(收入|利润|现金|资产|负债|权益|收益|费用|净额|金额|增长|同比|余额|每股|毛利|应收|应付|投资|revenue|profit|cash|asset|liabilit)", term, re.IGNORECASE)
                or term in quoted_terms(query_text)
            ]
            metric_hits = [term for term in metric_terms if self._contains_term(combined_text, term)]
            if metric_hits:
                utility += min(len(metric_hits), 3) * 6
                hits.append(f"metric_hit={metric_hits[:3]}")

            title_hits = [term for term in self._title_terms(doc_title) if self._contains_term(query_text, term)]
            if title_hits:
                utility += min(len(title_hits), 2) * 8
                hits.append(f"company_title_hit={title_hits[:2]}")

            table_signal = anchor.get("anchor_type") == "table_region" or "|" in combined_text
            numeric_signal = bool(re.search(r"\d", combined_text))
            if table_signal and numeric_signal:
                utility += 7
                hits.append("numeric_table_context")
            if metric_hits and numeric_signal:
                utility += 5
                hits.append("metric_numeric_cooccurrence")

            query_entities = self._query_entity_terms(query_text)
            if query_entities and not any(self._contains_term(doc_title, term) for term in query_entities):
                utility -= 5
                hits.append(f"wrong_company_penalty={query_entities[:2]}")

        elif record_type == "paper":
            direction = self._paper_direction(query_text)
            target_terms = [
                term
                for term in quoted_terms(query_text)
                if len(term) > 3 and term.casefold() not in {"reference", "references", "citation", "citations"}
            ]
            target_in_title = any(self._contains_term(doc_title, term) for term in target_terms)
            target_in_anchor = any(self._contains_term(combined_text, term) for term in target_terms)
            reference_section = bool(re.search(r"(reference|bibliography|参考文献)", text_lower))
            citation_cue = bool(re.search(r"(citation|cited|et al\.|\[[0-9]+\]|引用|参考文献)", text_lower))

            if direction in {"reference", "both"}:
                if target_in_title:
                    utility += 10
                    hits.append("target_doc_for_reference")
                if reference_section:
                    utility += 7
                    hits.append("reference_section")
                if target_in_title and reference_section:
                    utility += 5
                    hits.append("target_reference_section")
                if target_terms and not target_in_title and target_in_anchor and reference_section:
                    utility -= 4
                    hits.append("likely_reverse_citation_penalty")

            if direction in {"citation", "both"}:
                if target_terms and target_in_anchor and not target_in_title:
                    utility += 12
                    hits.append("other_doc_cites_target")
                elif target_in_anchor:
                    utility += 5
                    hits.append("target_mentioned")
                if reference_section and target_in_anchor and not target_in_title:
                    utility += 7
                    hits.append("citation_direction_reference_list")
                if target_in_title and reference_section and direction == "citation":
                    utility -= 4
                    hits.append("target_own_references_penalty")

            if citation_cue:
                utility += 3
                hits.append("citation_format_cue")

        elif record_type == "legal":
            important_terms = self._important_terms(query_text)
            label_hits = [
                term
                for term in important_terms
                if self._contains_term(combined_text, term)
                and re.search(r"(案由|判决|裁定|结果|罪|纠纷|撤销|驳回|赔偿|行政|刑事|民事)", term)
            ]
            if label_hits:
                utility += min(len(label_hits), 3) * 5
                hits.append(f"legal_label_hit={label_hits[:3]}")
            title_hits = [term for term in self._title_terms(doc_title) if self._contains_term(query_text, term)]
            if title_hits:
                utility += min(len(title_hits), 2) * 5
                hits.append(f"case_title_hit={title_hits[:2]}")
            if re.search(r"(本院认为|经审理查明|裁判|判决结果|案由|撤销|驳回|罪|纠纷|行政|刑事|民事)", text_lower):
                utility += 6
                hits.append("legal_decision_context")

        return utility, hits

    def _coverage_score(
        self,
        *,
        anchor: Dict[str, Any],
        combined_text: str,
        coverage_hints: Dict[str, Any],
        avoid_anchor_ids: List[str],
    ) -> Tuple[float, float, List[str], List[str]]:
        anchor_id = str(anchor.get("anchor_id", ""))
        text_lower = combined_text.casefold()
        coverage = 0.0
        redundancy_penalty = 0.0
        hits: List[str] = []
        redundancy_hits: List[str] = []

        missing_terms = []
        for key in ("missing_answer_keys", "low_confidence_keys", "must_find"):
            value = coverage_hints.get(key, [])
            if isinstance(value, list):
                missing_terms.extend(normalize_ws(item) for item in value if normalize_ws(item))
        term_hits = [term for term in missing_terms if self._contains_term(combined_text, term)]
        if term_hits:
            coverage += min(len(term_hits), 4) * 3
            hits.append(f"missing_slot_hit={term_hits[:4]}")

        if coverage_hints.get("needs_more_evidence") and len(normalize_ws(anchor.get("text", ""))) > 80:
            coverage += 2
            hits.append("needs_more_evidence")

        if coverage_hints.get("needs_non_doc_placeholder") and not re.fullmatch(r"DOC\d+", normalize_ws(anchor.get("summary", ""))):
            coverage += 2
            hits.append("non_doc_placeholder_candidate")

        selected_ids = set(avoid_anchor_ids)
        selected_ids.update(coverage_hints.get("selected_anchor_ids", []) or [])
        if anchor_id in selected_ids:
            redundancy_penalty += 40
            redundancy_hits.append("already_selected_anchor")
        else:
            coverage += 2
            hits.append("novel_anchor")

        known_values = coverage_hints.get("known_value_signatures", []) or []
        repeated_values = [
            normalize_ws(value)
            for value in known_values
            if normalize_ws(value) and normalize_ws(value).casefold() in text_lower
        ]
        if repeated_values:
            redundancy_penalty += min(len(repeated_values), 3) * 3
            redundancy_hits.append(f"repeats_known_value={repeated_values[:3]}")

        anchor_signature = f"{anchor.get('section_path', '')} {anchor.get('summary', '')} {compact_text(anchor.get('text', ''), limit=500)}"
        similarities = [
            self._text_similarity(anchor_signature, visited)
            for visited in coverage_hints.get("visited_anchor_texts", []) or []
            if normalize_ws(visited)
        ]
        max_similarity = max(similarities) if similarities else 0.0
        if max_similarity >= 0.62:
            penalty = max_similarity * 10
            redundancy_penalty += penalty
            redundancy_hits.append(f"similar_to_visited={max_similarity:.2f}")

        return coverage, redundancy_penalty, hits, redundancy_hits

    def _score_anchor(
        self,
        *,
        record: Dict[str, Any],
        doc_title: str,
        current_query: str,
        anchor: Dict[str, Any],
        coverage_hints: Optional[Dict[str, Any]] = None,
        avoid_anchor_ids: Optional[List[str]] = None,
    ) -> Tuple[float, List[str], Dict[str, Any]]:
        record_type = str(record.get("type", "")).strip().lower()
        level = int(record.get("level", 0) or 0)
        summary = str(anchor.get("summary", ""))
        section_path = str(anchor.get("section_path", ""))
        anchor_type = str(anchor.get("anchor_type", ""))
        anchor_text = str(anchor.get("text", ""))
        combined = f"{doc_title} {section_path} {anchor_type} {summary} {compact_text(anchor_text, limit=2200)}"
        combined_lower = combined.casefold()

        query_tokens = tokenize_query(current_query)
        instruction_tokens = tokenize_query(str(record.get("instruction", "")))
        quote_tokens = quoted_terms(str(record.get("question", "")) + "\n" + str(record.get("instruction", "")))

        local_score = 0.0
        prior_score = 0.0
        reasons: List[str] = []

        overlap = sum(1 for token in query_tokens if token in combined_lower)
        if overlap:
            local_score += overlap * 4
            reasons.append(f"query_overlap={overlap}")

        instruction_overlap = sum(1 for token in instruction_tokens[:12] if token in combined_lower)
        if instruction_overlap:
            local_score += min(instruction_overlap, 4) * 2
            reasons.append(f"instruction_overlap={instruction_overlap}")

        quote_hits = sum(1 for token in quote_tokens if token.casefold() in combined_lower)
        if quote_hits:
            local_score += quote_hits * 6
            reasons.append(f"quoted_hit={quote_hits}")

        if any(re.search(r"\d", token) for token in query_tokens) and re.search(r"\d", combined):
            local_score += 3
            reasons.append("numeric_match")

        section_score = self._section_prior(record_type, level, section_path)
        if section_score:
            prior_score += section_score * 0.35
            reasons.append(f"section_prior={section_score}")

        anchor_type_score = self._anchor_type_prior(record_type, level, anchor_type)
        if anchor_type_score:
            prior_score += anchor_type_score * 0.35
            reasons.append(f"anchor_type_prior={anchor_type_score}")

        task_score = self._task_prior(level, summary, section_path)
        if task_score:
            prior_score += task_score * 0.35
            reasons.append(f"task_prior={task_score}")

        if record_type == "paper" and re.search(r"(reference|citation|et al\.|\[[0-9]+\])", combined_lower):
            prior_score += 3
            reasons.append("citation_cue")

        if record_type == "financial" and re.search(r"(元|%|revenue|profit|cash|资产|利润|收入|应收)", combined_lower):
            prior_score += 2
            reasons.append("financial_cue")

        if record_type == "legal" and re.search(r"(案由|判决|裁定|撤销|驳回|罪|纠纷|行政)", combined_lower):
            prior_score += 2
            reasons.append("legal_cue")

        position_bonus = max(0.0, 2.5 - math.log(anchor.get("order", 1) + 1))
        prior_score += min(position_bonus, 1.5)
        reasons.append(f"position_bonus={position_bonus:.2f}")

        utility_score, utility_hits = self._task_utility_score(
            record=record,
            doc_title=doc_title,
            current_query=current_query,
            anchor=anchor,
            combined_text=combined,
        )
        if utility_hits:
            reasons.extend(utility_hits)

        coverage_hints = coverage_hints or {}
        coverage_score, redundancy_penalty, coverage_hits, redundancy_hits = self._coverage_score(
            anchor=anchor,
            combined_text=combined,
            coverage_hints=coverage_hints,
            avoid_anchor_ids=avoid_anchor_ids or [],
        )
        if coverage_hits:
            reasons.extend(coverage_hits)
        if redundancy_hits:
            reasons.extend(redundancy_hits)

        score = local_score + utility_score + coverage_score + prior_score - redundancy_penalty
        debug = {
            "utility_score": round(utility_score, 3),
            "utility_hits": utility_hits,
            "coverage_hints": coverage_hints,
            "score_breakdown": {
                "local_relevance": round(local_score, 3),
                "answer_utility": round(utility_score, 3),
                "coverage": round(coverage_score, 3),
                "weak_priors": round(prior_score, 3),
                "redundancy_penalty": round(redundancy_penalty, 3),
                "total": round(score, 3),
            },
        }
        return score, reasons, debug

    def _score_all_anchors(
        self,
        *,
        record: Dict[str, Any],
        doc_payload: Dict[str, Any],
        current_query: str,
        remaining_anchor_ids: Optional[List[str]] = None,
        coverage_hints: Optional[Dict[str, Any]] = None,
        avoid_anchor_ids: Optional[List[str]] = None,
    ) -> List[Tuple[float, Dict[str, Any], List[str], Dict[str, Any]]]:
        allowed = set(remaining_anchor_ids) if remaining_anchor_ids else None
        scored: List[Tuple[float, Dict[str, Any], List[str], Dict[str, Any]]] = []
        for anchor in doc_payload.get("anchors", []):
            if allowed is not None and anchor["anchor_id"] not in allowed:
                continue
            score, reasons, debug = self._score_anchor(
                record=record,
                doc_title=doc_payload["doc_title"],
                current_query=current_query,
                anchor=anchor,
                coverage_hints=coverage_hints,
                avoid_anchor_ids=avoid_anchor_ids,
            )
            scored.append((score, anchor, reasons, debug))
        scored.sort(key=lambda item: (item[0], -int(item[1].get("order", 0))), reverse=True)
        return scored

    @staticmethod
    def _chunk_summary_text(scored_anchors: List[Tuple[float, Dict[str, Any], List[str], Dict[str, Any]]]) -> str:
        blocks: List[str] = []
        for _, anchor, _, _ in scored_anchors:
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
        coverage_hints: Optional[Dict[str, Any]] = None,
        avoid_anchor_ids: Optional[List[str]] = None,
        round_index: int = 1,
    ) -> Dict[str, Any]:
        base_query = current_query_override or current_query_from_record(
            str(record.get("question", "")).strip(),
            str(record.get("instruction", "")).strip(),
        )
        fallback = self._heuristic_plan(record, doc_payload, base_query)
        if must_find_override:
            fallback["must_find"] = [normalize_ws(item) for item in must_find_override if normalize_ws(item)]

        known_items = known_items or []
        merged_coverage_hints = dict(self._known_item_hints(known_items))
        if coverage_hints:
            merged_coverage_hints.update(coverage_hints)
        if fallback["must_find"] and "must_find" not in merged_coverage_hints:
            merged_coverage_hints["must_find"] = fallback["must_find"]
        avoid_anchor_ids = [normalize_ws(item) for item in (avoid_anchor_ids or []) if normalize_ws(item)]

        scored = self._score_all_anchors(
            record=record,
            doc_payload=doc_payload,
            current_query=fallback["current_query"],
            remaining_anchor_ids=remaining_anchor_ids,
            coverage_hints=merged_coverage_hints,
            avoid_anchor_ids=avoid_anchor_ids,
        )
        shortlist = scored[: self.max_catalog_anchors]
        default_next_anchor_id = shortlist[0][1]["anchor_id"] if shortlist else ""

        result = {
            "think": fallback["think"],
            "current_query": fallback["current_query"],
            "must_find": fallback["must_find"],
            "next_anchor_id": default_next_anchor_id,
            "candidate_anchor_ids": [anchor["anchor_id"] for _, anchor, _, _ in shortlist[:4]],
            "ranked_anchor_ids": [anchor["anchor_id"] for _, anchor, _, _ in scored[:4]],
            "all_ranked_anchor_ids": [anchor["anchor_id"] for _, anchor, _, _ in scored],
            "rule_hits": [
                {
                    "anchor_id": anchor["anchor_id"],
                    "score": round(score, 3),
                    "reasons": reasons,
                    "utility_score": debug.get("utility_score", 0.0),
                    "utility_hits": debug.get("utility_hits", []),
                    "score_breakdown": debug.get("score_breakdown", {}),
                    "summary": compact_text(anchor.get("summary", ""), limit=180),
                }
                for score, anchor, reasons, debug in scored[:4]
            ],
            "utility_hits": {
                anchor["anchor_id"]: debug.get("utility_hits", [])
                for _, anchor, _, debug in scored[:4]
                if debug.get("utility_hits")
            },
            "utility_score": {
                anchor["anchor_id"]: debug.get("utility_score", 0.0)
                for _, anchor, _, debug in scored[:4]
            },
            "coverage_hints": merged_coverage_hints,
            "avoid_anchor_ids": avoid_anchor_ids,
            "score_breakdown": {
                anchor["anchor_id"]: debug.get("score_breakdown", {})
                for _, anchor, _, debug in scored[:4]
            },
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
                coverage_hints=merged_coverage_hints,
                avoid_anchor_ids=avoid_anchor_ids,
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
            valid_ids = {anchor["anchor_id"] for _, anchor, _, _ in rescored}
            if next_anchor_id in valid_ids:
                result["next_anchor_id"] = next_anchor_id
            elif requested_ids:
                for anchor_id in requested_ids:
                    if anchor_id in valid_ids:
                        result["next_anchor_id"] = anchor_id
                        break
            elif shortlist:
                result["next_anchor_id"] = shortlist[0][1]["anchor_id"]
            result["candidate_anchor_ids"] = requested_ids[:4] or [anchor["anchor_id"] for _, anchor, _, _ in shortlist[:4]]
            result["ranked_anchor_ids"] = [anchor["anchor_id"] for _, anchor, _, _ in rescored[:4]]
            result["all_ranked_anchor_ids"] = [anchor["anchor_id"] for _, anchor, _, _ in rescored]
            result["rule_hits"] = [
                {
                    "anchor_id": anchor["anchor_id"],
                    "score": round(score, 3),
                    "reasons": reasons,
                    "utility_score": debug.get("utility_score", 0.0),
                    "utility_hits": debug.get("utility_hits", []),
                    "score_breakdown": debug.get("score_breakdown", {}),
                    "summary": compact_text(anchor.get("summary", ""), limit=180),
                }
                for score, anchor, reasons, debug in rescored[:4]
            ]
            result["utility_hits"] = {
                anchor["anchor_id"]: debug.get("utility_hits", [])
                for _, anchor, _, debug in rescored[:4]
                if debug.get("utility_hits")
            }
            result["utility_score"] = {
                anchor["anchor_id"]: debug.get("utility_score", 0.0)
                for _, anchor, _, debug in rescored[:4]
            }
            result["score_breakdown"] = {
                anchor["anchor_id"]: debug.get("score_breakdown", {})
                for _, anchor, _, debug in rescored[:4]
            }
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
            "coverage_hints": decision.get("coverage_hints", {}),
            "avoid_anchor_ids": decision.get("avoid_anchor_ids", []),
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
            "utility_hits": decision.get("utility_hits", {}),
            "utility_score": decision.get("utility_score", {}),
            "coverage_hints": decision.get("coverage_hints", {}),
            "avoid_anchor_ids": decision.get("avoid_anchor_ids", []),
            "score_breakdown": decision.get("score_breakdown", {}),
            "why_these_anchors": why,
            "raw_plan_text": decision.get("raw_plan_text", ""),
            "tagged_trace": tagged_trace,
        }
        write_json(cache_path, payload)
        return payload
