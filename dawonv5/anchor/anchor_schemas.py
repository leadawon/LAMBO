"""dawonv5 enriched anchor metadata schema.

Extends the dawonv4 anchor record with richer structured metadata fields.
All new fields are Optional — existing code that doesn't know about them
continues to work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CitationEdge:
    """A directed cross-reference edge between two anchors."""
    source_anchor_id: str
    target_anchor_id: str
    relation_type: str  # explains_table, cites_previous, elaborates_heading, references_figure, etc.
    evidence_snippet: str = ""
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CitationEdge":
        return cls(
            source_anchor_id=d.get("source_anchor_id", ""),
            target_anchor_id=d.get("target_anchor_id", ""),
            relation_type=d.get("relation_type", "unknown"),
            evidence_snippet=d.get("evidence_snippet", ""),
            confidence=d.get("confidence", 0.5),
        )


@dataclass
class AnchorEnrichedMetadata:
    """Enriched metadata for a single anchor (dawonv5).

    These fields are layered ON TOP of dawonv4's existing annotation fields
    (anchor_role_candidates, anchor_entities, anchor_targets, anchor_value_hints,
    provenance_hints, anchor_relations, relation_summary).
    """

    # --- Identity / affiliation ---
    anchor_owner_type: str = "unknown"
    # company | court_case | paper | regulation | contract | article | report | unknown
    anchor_owner_name: str = ""
    anchor_parent_heading: str = ""

    # --- Content characterization ---
    anchor_content_type: str = "paragraph"
    # paragraph | table | figure | list | quote | footnote | mixed
    anchor_semantic_role: str = "unknown"
    # definition | claim | evidence | statistic | comparison | methodology | result
    # table_summary | figure_summary | case_fact | holding | legal_argument
    # citation_context | procedure | unknown

    # --- Temporal & numeric ---
    anchor_time_scope: str = ""
    anchor_unit_hints: List[str] = field(default_factory=list)

    # --- Cross-reference edges (from citation graph) ---
    citation_edges_out: List[Dict[str, Any]] = field(default_factory=list)
    citation_edges_in: List[Dict[str, Any]] = field(default_factory=list)
    citation_summary: str = ""

    # --- Provenance ---
    v5_provenance_flags: List[str] = field(default_factory=list)
    v5_confidence: float = 0.5

    # --- Rendered summary ---
    rendered_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnchorEnrichedMetadata":
        if not d:
            return cls()
        kw = {}
        for f in cls.__dataclass_fields__:
            if f in d:
                kw[f] = d[f]
        return cls(**kw)

    def merge_provenance(self, flag: str) -> None:
        if flag not in self.v5_provenance_flags:
            self.v5_provenance_flags.append(flag)
