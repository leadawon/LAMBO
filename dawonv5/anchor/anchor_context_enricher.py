"""dawonv5 anchor context enricher.

Extracts owner type/name, content type, semantic role, time scope,
unit hints from anchor records. Works on TOP of dawonv4's existing
annotate_anchor() fields (anchor_role_candidates, anchor_entities, etc.).

Rule-based first; optional LLM fallback for ambiguous semantic roles.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .anchor_schemas import AnchorEnrichedMetadata
from .common import compact_text, normalize_ws

# ---------------------------------------------------------------------------
# Owner type patterns (applied to doc_title)
# ---------------------------------------------------------------------------

_OWNER_PATTERNS: List[Dict[str, Any]] = [
    # Chinese court cases
    {"regex": r"([\u4e00-\u9fff]+(?:法院|仲裁委)).*?(\d{4}[\u4e00-\u9fff]*\d+号)", "type": "court_case"},
    {"regex": r"判决文书|裁定书|裁判文书|判决书", "type": "court_case"},
    # Chinese financial reports
    {"regex": r"([\u4e00-\u9fff]+(?:股份|集团|公司)[\u4e00-\u9fff]*).*?((?:年度|半年度|季度)报告|事业报告书|招股说明书)", "type": "company"},
    {"regex": r"证券简称[:：]\s*([^\s|　]+)", "type": "company"},
    {"regex": r"证券代码[:：]\s*(\d{6})", "type": "company"},
    # Paper patterns
    {"regex": r"\b(arXiv|doi|et al\.|proceedings|journal|Abstract)\b", "type": "paper", "flags": re.IGNORECASE},
    # Regulation
    {"regex": r"([\u4e00-\u9fff]+(?:法|条例|规定|办法|准则|规则))$", "type": "regulation"},
    # Contract
    {"regex": r"(合同|协议|契约|agreement|contract)", "type": "contract", "flags": re.IGNORECASE},
]

# ---------------------------------------------------------------------------
# Semantic role heuristics
# ---------------------------------------------------------------------------

_ROLE_KEYWORDS: List[Dict[str, Any]] = [
    {"pattern": r"(定义|definition|是指|means|refers to)", "role": "definition"},
    {"pattern": r"(本院认为|court holds|holding|认为.*应当)", "role": "holding"},
    {"pattern": r"(经审理查明|事实认定|facts found|案件事实)", "role": "case_fact"},
    {"pattern": r"(原告|被告|上诉|plaintiff|defendant|诉讼请求)", "role": "legal_argument"},
    {"pattern": r"(方法|methodology|approach|algorithm|model architecture)", "role": "methodology"},
    {"pattern": r"(实验|experiment|evaluation|benchmark|ablation)", "role": "result"},
    {"pattern": r"(结果|result|performance|accuracy|precision|recall|f1)", "role": "result"},
    {"pattern": r"(比较|comparison|compared to|versus|vs\.?|相比)", "role": "comparison"},
    {"pattern": r"(引用|citation|reference|bibliography|参考文献|\[\d+\])", "role": "citation_context"},
    {"pattern": r"(evidence|证据|证明|表明|demonstrates|shows that)", "role": "evidence"},
]

# ---------------------------------------------------------------------------
# Time scope regex
# ---------------------------------------------------------------------------

_TIME_PATTERNS = [
    re.compile(r"((?:FY|CY)?\d{4})\s*(?:年)?\s*(?:第?([一二三四1-4])季度|Q([1-4]))", re.IGNORECASE),
    re.compile(r"(\d{4})\s*[~\-–至到]\s*(\d{4})"),
    re.compile(r"(\d{4})\s*(?:年度|年|회계연도)"),
]

_QUARTER_MAP = {"一": "Q1", "二": "Q2", "三": "Q3", "四": "Q4", "1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}

# ---------------------------------------------------------------------------
# Unit hints regex
# ---------------------------------------------------------------------------

_UNIT_PATTERNS = [
    re.compile(r"(百万|万|亿|千)\s*(?:元|美元|韩元|日元|港元|欧元|英镑)"),
    re.compile(r"(?:单位[:：]\s*)([\u4e00-\u9fff]+)"),
    re.compile(r"(元|美元|韩元|USD|CNY|KRW|EUR|GBP)", re.IGNORECASE),
    re.compile(r"(%|百分比|百分点|bps|basis points)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Content type inference
# ---------------------------------------------------------------------------

def _infer_content_type(text: str) -> str:
    """Infer content type from anchor text patterns."""
    if text.count("|") >= 4:
        return "table"
    if re.search(r"(图|figure|fig\.)\s*\d+", text[:300], re.IGNORECASE):
        return "figure"
    if re.match(r"^\s*[-*•]\s+", text) or re.match(r"^\s*\d+[.)]\s+", text):
        return "list"
    if re.match(r"^\s*>", text) or re.search(r"(引述|引用|quote)", text[:100], re.IGNORECASE):
        return "quote"
    return "paragraph"


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def _extract_owner(doc_title: str, record_type: str) -> Dict[str, str]:
    """Extract owner_type and owner_name from doc title."""
    type_hint_map = {"financial": "company", "legal": "court_case", "paper": "paper"}
    default_type = type_hint_map.get(record_type, "unknown")
    owner_name = ""
    owner_type = default_type

    for pat in _OWNER_PATTERNS:
        flags = pat.get("flags", 0)
        m = re.search(pat["regex"], doc_title, flags=flags)
        if m:
            owner_type = pat["type"]
            owner_name = next((g for g in m.groups() if g), "") if m.lastindex else ""
            break

    if not owner_name:
        # Clean doc_title as fallback name
        owner_name = re.sub(r"[《》「」]", "", doc_title).strip()

    return {"owner_type": owner_type, "owner_name": owner_name}


def _extract_time_scope(text: str) -> str:
    """Extract normalized time scope from text."""
    for pat in _TIME_PATTERNS:
        m = pat.search(text[:600])
        if m:
            groups = m.groups()
            if len(groups) >= 3 and (groups[1] or groups[2]):
                year = groups[0]
                quarter = _QUARTER_MAP.get(groups[1] or groups[2] or "", "")
                return f"{year}{quarter}"
            elif len(groups) == 2 and groups[1]:
                return f"{groups[0]}-{groups[1]}"
            else:
                return groups[0]
    return ""


def _extract_unit_hints(text: str) -> List[str]:
    """Extract unit hints from text."""
    hints = []
    seen = set()
    for pat in _UNIT_PATTERNS:
        for m in pat.finditer(text[:1200]):
            hint = m.group(0).strip()
            if hint and hint not in seen:
                seen.add(hint)
                hints.append(hint)
    return hints[:5]


def _classify_semantic_role(
    text: str,
    anchor_title: str,
    record_type: str,
    v4_roles: List[str],
) -> str:
    """Classify semantic role using keyword heuristics + v4 role hints.

    Leverages dawonv4's anchor_role_candidates to boost confidence.
    """
    # Use content type as fast signal
    if text.count("|") >= 4:
        if re.search(r"(비교|比较|comparison|vs)", text[:300], re.IGNORECASE):
            return "comparison"
        return "statistic"

    # v4 roles can inform v5 semantic roles
    v4_set = set(v4_roles)
    if "decision_evidence" in v4_set:
        return "holding"
    if "case_identity" in v4_set:
        return "case_fact"
    if "citation_direction" in v4_set:
        return "citation_context"

    combined = f"{anchor_title}\n{text[:800]}"
    best_role = "unknown"
    best_count = 0
    for entry in _ROLE_KEYWORDS:
        count = len(re.findall(entry["pattern"], combined, re.IGNORECASE))
        if count > best_count:
            best_count = count
            best_role = entry["role"]

    return best_role


def _extract_parent_heading(anchor_title: str, doc_title: str) -> str:
    """Best guess at parent heading from anchor_title and doc_title."""
    if anchor_title and anchor_title.lower() not in {"preamble", "document", "anchor"}:
        return anchor_title
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_anchor(
    anchor: Dict[str, Any],
    *,
    record_type: str = "",
    doc_title: str = "",
    use_llm: bool = False,
    llm: Any = None,
) -> AnchorEnrichedMetadata:
    """Enrich a single anchor record with dawonv5 metadata.

    Designed to run AFTER dawonv4's annotate_anchor() has already populated
    anchor_role_candidates, anchor_entities, anchor_targets, etc.
    """
    text = anchor.get("text", "")
    anchor_title = anchor.get("anchor_title", "")
    doc_title = doc_title or anchor.get("doc_title", "")
    v4_roles = anchor.get("anchor_role_candidates", [])

    provenance: List[str] = []

    # Owner
    owner_info = _extract_owner(doc_title, record_type)
    provenance.append("heading-derived")

    # Content type
    content_type = _infer_content_type(text)

    # Semantic role (leverages v4 role candidates)
    semantic_role = _classify_semantic_role(text, anchor_title, record_type, v4_roles)
    provenance.append("regex-derived")

    # Time scope
    time_scope = _extract_time_scope(doc_title + "\n" + text)

    # Unit hints
    unit_hints = _extract_unit_hints(text)

    # Parent heading
    parent_heading = _extract_parent_heading(anchor_title, doc_title)

    # Confidence
    filled = sum([
        bool(owner_info["owner_name"]),
        semantic_role != "unknown",
        bool(time_scope),
        bool(unit_hints),
        bool(anchor.get("anchor_entities")),
    ])
    confidence = min(0.3 + filled * 0.14, 1.0)

    # Optional LLM for semantic_role
    if use_llm and llm and semantic_role == "unknown":
        try:
            llm_role = _llm_classify_role(llm, text[:600], anchor_title, record_type)
            if llm_role and llm_role != "unknown":
                semantic_role = llm_role
                provenance.append("llm-derived")
                confidence = min(confidence + 0.1, 1.0)
        except Exception:
            pass

    return AnchorEnrichedMetadata(
        anchor_owner_type=owner_info["owner_type"],
        anchor_owner_name=owner_info["owner_name"],
        anchor_parent_heading=parent_heading,
        anchor_content_type=content_type,
        anchor_semantic_role=semantic_role,
        anchor_time_scope=time_scope,
        anchor_unit_hints=unit_hints,
        v5_provenance_flags=provenance,
        v5_confidence=round(confidence, 2),
    )


def enrich_anchors(
    anchors: List[Dict[str, Any]],
    *,
    record_type: str = "",
    doc_title: str = "",
    use_llm: bool = False,
    llm: Any = None,
) -> List[AnchorEnrichedMetadata]:
    """Enrich all anchors in a document."""
    return [
        enrich_anchor(a, record_type=record_type, doc_title=doc_title, use_llm=use_llm, llm=llm)
        for a in anchors
    ]


def _llm_classify_role(llm: Any, text_snippet: str, anchor_title: str, record_type: str) -> str:
    """LLM fallback for semantic role classification."""
    system = "You classify text segments into semantic roles. Reply with exactly one role label."
    roles = "definition|claim|evidence|statistic|comparison|methodology|result|table_summary|figure_summary|case_fact|holding|legal_argument|citation_context|procedure|unknown"
    user = (
        f"Document type: {record_type}\n"
        f"Anchor title: {anchor_title}\n"
        f"Text: {text_snippet}\n\n"
        f"Classify into one of: {roles}\n"
        f"Reply with the role label only."
    )
    raw = llm.generate_text(
        system_prompt=system,
        user_prompt=user,
        max_output_tokens=20,
        metadata={"module": "v5_enricher_role"},
    )
    role = raw.strip().lower().replace(" ", "_")
    valid = set(roles.split("|"))
    return role if role in valid else "unknown"
