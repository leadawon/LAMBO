"""dawonv5 anchor summary renderer.

Converts structured AnchorEnrichedMetadata + dawonv4 anchor record
into a concise, retrieval-friendly rendered summary (≤2 sentences).
"""
from __future__ import annotations

from typing import Any, Dict, List

from .schemas import AnchorEnrichedMetadata
from lambo.common import compact_text

MAX_SUMMARY_LEN = 300

_ROLE_LABELS = {
    "definition": "정의",
    "claim": "주장",
    "evidence": "근거",
    "statistic": "통계",
    "comparison": "비교",
    "methodology": "방법론",
    "result": "결과",
    "table_summary": "표 요약",
    "figure_summary": "그림 요약",
    "case_fact": "사실관계",
    "holding": "법원 판단",
    "legal_argument": "법적 논증",
    "citation_context": "인용 맥락",
    "procedure": "절차",
}

_CONTENT_LABELS = {
    "paragraph": "문단",
    "table": "표",
    "figure": "그림",
    "list": "목록",
    "quote": "인용",
    "footnote": "각주",
    "mixed": "혼합 영역",
}

_OWNER_TYPE_LABELS = {
    "company": "기업 공시",
    "court_case": "판결문",
    "paper": "논문",
    "regulation": "법령",
    "contract": "계약서",
    "article": "기사",
    "report": "보고서",
    "unknown": "문서",
}


def render_summary(
    meta: AnchorEnrichedMetadata,
    anchor: Dict[str, Any],
    *,
    original_summary: str = "",
) -> str:
    """Render a concise summary from structured metadata + v4 anchor.

    Parameters
    ----------
    meta : AnchorEnrichedMetadata
    anchor : dict  — the raw dawonv4 anchor record
    original_summary : str — the v4 LLM-generated summary

    Returns
    -------
    str — rendered summary ≤ MAX_SUMMARY_LEN chars
    """
    parts = []

    # 1) Identity line: owner + heading + role + content type
    owner_name = meta.anchor_owner_name
    if len(owner_name) > 50:
        owner_name = owner_name[:47] + "..."

    role_label = _ROLE_LABELS.get(meta.anchor_semantic_role, "")
    ct_label = _CONTENT_LABELS.get(meta.anchor_content_type, "영역")

    identity_parts = []
    if owner_name:
        identity_parts.append(owner_name)
    if meta.anchor_parent_heading:
        identity_parts.append(f"'{meta.anchor_parent_heading}'")

    if identity_parts:
        identity = " > ".join(identity_parts)
        if role_label:
            parts.append(f"{identity}의 {role_label} {ct_label}.")
        else:
            parts.append(f"{identity}의 {ct_label}.")

    # 2) Core content from v4 summary
    core = original_summary or anchor.get("summary", "")
    if core:
        core = compact_text(core, limit=120)
        parts.append(core)

    # 3) Metadata tail (time, units, entities, citations)
    tail_items = []
    if meta.anchor_time_scope:
        tail_items.append(meta.anchor_time_scope)
    if meta.anchor_unit_hints:
        tail_items.append("/".join(meta.anchor_unit_hints[:2]))

    # Use v4 entities if richer
    entities = anchor.get("anchor_entities", []) or anchor.get("key_entities", [])
    if entities:
        tail_items.append(", ".join(entities[:2]))

    if meta.citation_summary:
        cs = meta.citation_summary
        if len(cs) > 50:
            cs = cs[:47] + "..."
        tail_items.append(cs)

    if tail_items:
        parts.append(f"[{' | '.join(tail_items)}]")

    rendered = " ".join(parts).strip()
    if len(rendered) > MAX_SUMMARY_LEN:
        rendered = rendered[: MAX_SUMMARY_LEN - 3].rstrip() + "..."
    return rendered


def render_summaries_for_doc(
    anchors: List[Dict[str, Any]],
    enrichments: List[AnchorEnrichedMetadata],
) -> List[str]:
    """Render summaries for all anchors in a document.

    Returns list of rendered summary strings.
    Also sets each enrichment's rendered_summary field.
    """
    results = []
    for i, anchor in enumerate(anchors):
        meta = enrichments[i] if i < len(enrichments) else AnchorEnrichedMetadata()
        original = anchor.get("summary", "")
        rendered = render_summary(meta, anchor, original_summary=original)
        meta.rendered_summary = rendered
        results.append(rendered)
    return results
