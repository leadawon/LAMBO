"""dawonv8 DocRefineAgent — show v5 rendered summaries in the doc map.

The base DocRefineAgent presents every anchor's `summary` to the LLM so it
can decide which to open. With v5 enrichment the anchor record also carries
a richer `rendered_summary` plus semantic_role / content_type / time_scope /
owner — surfacing those in the doc map gives the LLM better retrieval signal
without changing the loop structure.
"""

from __future__ import annotations

from typing import Any, Dict, List

from lambo.agents.doc_refine_agent import DocRefineAgent
from lambo.common import compact_text


class EnrichedDocRefineAgent(DocRefineAgent):
    @staticmethod
    def _format_doc_map(anchors: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for anchor in anchors:
            v5 = anchor.get("v5_metadata") or {}
            summary = (
                anchor.get("rendered_summary")
                or anchor.get("summary", "")
            )
            tags: List[str] = []
            role = v5.get("anchor_semantic_role", "") or ""
            if role and role != "unknown":
                tags.append(role)
            content_type = v5.get("anchor_content_type", "") or ""
            if content_type and content_type != "paragraph":
                tags.append(content_type)
            time_scope = v5.get("anchor_time_scope", "") or ""
            if time_scope:
                tags.append(time_scope)
            citation_summary = v5.get("citation_summary", "") or ""

            tag_str = f" ({'/'.join(tags)})" if tags else ""
            line = (
                f"[{anchor['anchor_id']}]{tag_str} {anchor.get('anchor_title', '')} "
                f"| {compact_text(summary, limit=220)}"
            )
            if citation_summary:
                line += f" | refs: {compact_text(citation_summary, limit=80)}"
            lines.append(line)
        return "\n".join(lines) if lines else "(no anchors available)"
