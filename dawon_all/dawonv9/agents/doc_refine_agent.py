"""dawonv9 DocRefineAgent — v8 enriched doc_map + cross-doc citation hints.

New in dawonv9
--------------
If Phase 2 of AnchorAgent injected ``cross_doc_cites`` into a reference-
section anchor, ``_format_doc_map`` now surfaces that as a prominent "★"
marker so the LLM immediately knows which anchor to open and which
documents it cites.

Example doc_map line (new):
  [DOC2_A7] (citation_context) References | ★ CITES: DOC1 (HuBERT), DOC3 (SUPERB) | refs: …

Without the hint the LLM often opened the anchor, saw "[23] HuBERT: How
much can a bad teacher…" and concluded there was "no evidence" because the
subtitle differed from DOC1's title.  The hint bypasses that judgement.
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

            # -- tags: semantic role, content type, time scope --
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
                f"| {compact_text(summary, limit=200)}"
            )

            # -- v9: cross-doc citation hint --
            cross_cites: List[Dict[str, Any]] = anchor.get("cross_doc_cites") or []
            if cross_cites:
                cited_parts = []
                for cc in cross_cites[:4]:
                    doc_id = cc.get("doc_id", "")
                    doc_title = cc.get("doc_title", "")
                    short_title = doc_title[:40] + ("…" if len(doc_title) > 40 else "")
                    cited_parts.append(f"{doc_id} ({short_title})")
                line += f" | ★ CITES: {', '.join(cited_parts)}"

            if citation_summary:
                line += f" | refs: {compact_text(citation_summary, limit=80)}"

            lines.append(line)
        return "\n".join(lines) if lines else "(no anchors available)"
