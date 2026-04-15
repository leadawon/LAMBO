"""dawonv5 citation and cross-reference graph extraction.

Builds on top of dawonv4's anchor_relations (which handles supports,
disambiguates, same_section, etc.) by adding EXPLICIT citation/cross-reference
edges that dawonv4 does not capture:
  1. Explicit citations ("Table 3", "Figure 2", "Section 4.1", "위 표", "supra")
  2. Local reference matching (anchor_title/text pattern → anchor_id)
  3. Structural adjacency (table↔explanation paragraph, heading→body)

These citation edges are stored separately from dawonv4's anchor_relations
to avoid breaking existing code.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .anchor_schemas import AnchorEnrichedMetadata, CitationEdge


# ---------------------------------------------------------------------------
# Citation regex patterns
# ---------------------------------------------------------------------------

_CITATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?:表|Table|table|Tab\.?)\s*(\d+)", re.IGNORECASE), "references_table"),
    (re.compile(r"(?:图|Figure|figure|Fig\.?)\s*(\d+)", re.IGNORECASE), "references_figure"),
    (re.compile(r"(?:Section|节|章)\s*(\d+(?:\.\d+)*)", re.IGNORECASE), "references_section"),
    (re.compile(r"(위\s*표|上表|上述|前述|전술한|前述した)", re.IGNORECASE), "cites_previous"),
    (re.compile(r"(아래\s*표|下表|下述|후술|後述する)", re.IGNORECASE), "cites_following"),
    (re.compile(r"(supra|앞서\s*언급|如前所述)", re.IGNORECASE), "cites_previous"),
    (re.compile(r"(infra|이하에서|如後所述)", re.IGNORECASE), "cites_following"),
    (re.compile(r"(?:注|脚注|footnote|fn\.?)\s*(\d+)", re.IGNORECASE), "references_footnote"),
    (re.compile(r"\[(\d{1,3})\]"), "cites_reference"),
]


def _find_explicit_citations(text: str) -> List[Tuple[str, str, str]]:
    """Return list of (relation_type, reference_key, evidence_snippet)."""
    results = []
    for pat, rel_type in _CITATION_PATTERNS:
        for m in pat.finditer(text[:2000]):
            ref_key = m.group(1) if m.lastindex else m.group(0)
            snippet = text[max(0, m.start() - 20): m.end() + 20].strip()
            results.append((rel_type, ref_key.strip(), snippet))
    return results


def _match_reference_to_anchor(
    ref_type: str,
    ref_key: str,
    anchors: List[Dict[str, Any]],
    source_anchor_id: str,
) -> Optional[str]:
    """Try to resolve a citation reference to a specific anchor_id."""
    ref_key_lower = ref_key.lower().strip()

    for anchor in anchors:
        if anchor["anchor_id"] == source_anchor_id:
            continue
        title = anchor.get("anchor_title", "").lower()
        text_start = anchor.get("text", "")[:150].lower()

        if ref_type == "references_table":
            if re.search(rf"(?:表|table)\s*{re.escape(ref_key_lower)}", text_start, re.IGNORECASE):
                return anchor["anchor_id"]
            if ref_key_lower in title and "|" in anchor.get("text", ""):
                return anchor["anchor_id"]

        elif ref_type == "references_figure":
            if re.search(rf"(?:图|figure|fig)\s*{re.escape(ref_key_lower)}", text_start, re.IGNORECASE):
                return anchor["anchor_id"]

        elif ref_type == "references_section":
            if ref_key_lower in title:
                return anchor["anchor_id"]

    return None


def _find_structural_edges(anchors: List[Dict[str, Any]]) -> List[CitationEdge]:
    """Find structural edges: table↔adjacent paragraph."""
    edges = []
    for i, anchor in enumerate(anchors):
        text = anchor.get("text", "")
        is_table = text.count("|") >= 4

        # Table followed by paragraph → paragraph elaborates table
        if is_table and i + 1 < len(anchors):
            next_a = anchors[i + 1]
            next_text = next_a.get("text", "")
            if next_text.count("|") < 4:
                edges.append(CitationEdge(
                    source_anchor_id=next_a["anchor_id"],
                    target_anchor_id=anchor["anchor_id"],
                    relation_type="elaborates_table",
                    confidence=0.6,
                ))

        # Paragraph followed by table → paragraph introduces table
        if not is_table and i + 1 < len(anchors):
            next_a = anchors[i + 1]
            next_text = next_a.get("text", "")
            if next_text.count("|") >= 4:
                edges.append(CitationEdge(
                    source_anchor_id=anchor["anchor_id"],
                    target_anchor_id=next_a["anchor_id"],
                    relation_type="introduces_table",
                    confidence=0.5,
                ))

    return edges


def build_citation_graph(
    anchors: List[Dict[str, Any]],
    enrichments: Optional[List[AnchorEnrichedMetadata]] = None,
) -> List[CitationEdge]:
    """Build citation edges for anchors within one document.

    These edges are SEPARATE from dawonv4's anchor_relations.
    """
    all_edges: List[CitationEdge] = []
    anchor_id_set = {a["anchor_id"] for a in anchors}

    # 1) Explicit citations
    for anchor in anchors:
        citations = _find_explicit_citations(anchor.get("text", ""))
        for rel_type, ref_key, snippet in citations:
            target_id = _match_reference_to_anchor(rel_type, ref_key, anchors, anchor["anchor_id"])
            if target_id and target_id in anchor_id_set:
                all_edges.append(CitationEdge(
                    source_anchor_id=anchor["anchor_id"],
                    target_anchor_id=target_id,
                    relation_type=rel_type,
                    evidence_snippet=snippet,
                    confidence=0.7,
                ))
            elif rel_type == "cites_previous":
                prev_id = anchor.get("prev_anchor_id", "")
                if prev_id and prev_id in anchor_id_set:
                    all_edges.append(CitationEdge(
                        source_anchor_id=anchor["anchor_id"],
                        target_anchor_id=prev_id,
                        relation_type="cites_previous",
                        evidence_snippet=snippet,
                        confidence=0.5,
                    ))
            elif rel_type == "cites_following":
                next_id = anchor.get("next_anchor_id", "")
                if next_id and next_id in anchor_id_set:
                    all_edges.append(CitationEdge(
                        source_anchor_id=anchor["anchor_id"],
                        target_anchor_id=next_id,
                        relation_type="cites_following",
                        evidence_snippet=snippet,
                        confidence=0.5,
                    ))

    # 2) Structural edges
    all_edges.extend(_find_structural_edges(anchors))

    # 3) Deduplicate
    seen: Set[Tuple[str, str, str]] = set()
    deduped = []
    for edge in all_edges:
        key = (edge.source_anchor_id, edge.target_anchor_id, edge.relation_type)
        if key not in seen:
            seen.add(key)
            deduped.append(edge)
    all_edges = deduped

    # 4) Attach to enrichments if provided
    if enrichments:
        enrich_map: Dict[str, AnchorEnrichedMetadata] = {}
        for i, anchor in enumerate(anchors):
            if i < len(enrichments):
                enrich_map[anchor["anchor_id"]] = enrichments[i]

        for edge in all_edges:
            if edge.source_anchor_id in enrich_map:
                enrich_map[edge.source_anchor_id].citation_edges_out.append({
                    "target": edge.target_anchor_id,
                    "type": edge.relation_type,
                })
            if edge.target_anchor_id in enrich_map:
                enrich_map[edge.target_anchor_id].citation_edges_in.append({
                    "source": edge.source_anchor_id,
                    "type": edge.relation_type,
                })

        # Build citation summaries
        for aid, em in enrich_map.items():
            parts = []
            for rel in em.citation_edges_out[:3]:
                parts.append(f"{aid} -> {rel['target']} : {rel['type']}")
            for rel in em.citation_edges_in[:3]:
                parts.append(f"{rel['source']} -> {aid} : {rel['type']}")
            em.citation_summary = "; ".join(parts) if parts else ""

    return all_edges


def edges_to_summary(edges: List[CitationEdge]) -> str:
    """Convert edge list to compact text summary."""
    if not edges:
        return ""
    return "\n".join(
        f"{e.source_anchor_id} -> {e.target_anchor_id} : {e.relation_type}"
        for e in edges[:20]
    )
