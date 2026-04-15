"""dawonv8 AnchorAgent — lambo AnchorAgent + v5 enrichment overlay.

After the base agent places anchors over each document, we run the v5
enrichment pipeline (owner/semantic_role/time/unit + citation graph +
rendered_summary) and fold the results into both the anchors and the
doc_map so downstream stages see richer context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from lambo.agents.anchor_agent import AnchorAgent
from lambo.common import (
    current_query_from_record,
    ensure_dir,
    instruction_from_record,
    parse_docs_bundle,
    read_json,
    safe_filename,
    write_json,
)

from ..enrichment import (
    build_citation_graph,
    enrich_anchors,
    render_summaries_for_doc,
)


class EnrichedAnchorAgent(AnchorAgent):
    """Base AnchorAgent with v5 enrichment applied per document."""

    def _enrich_doc_anchors(
        self,
        *,
        anchors: List[Dict[str, Any]],
        doc_title: str,
        record_type: str,
    ) -> None:
        if not anchors:
            return
        enrichments = enrich_anchors(
            anchors, record_type=record_type, doc_title=doc_title, use_llm=False
        )
        edges = build_citation_graph(anchors, enrichments)
        render_summaries_for_doc(anchors, enrichments)

        # Fold enrichment back into anchor records.
        for anchor, meta in zip(anchors, enrichments):
            meta_dict = meta.to_dict()
            anchor["v5_metadata"] = meta_dict
            if meta.rendered_summary:
                anchor["rendered_summary"] = meta.rendered_summary

        # Attach doc-level citation edge list (compact form).
        if edges and anchors:
            # Stash on the first anchor for convenience — also returned via
            # doc payload in run().
            anchors[0].setdefault("_doc_citation_edges", [e.to_dict() for e in edges])

    def run(
        self,
        *,
        record: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        ensure_dir(sample_dir)
        cache_path = sample_dir / "anchors.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        query = current_query_from_record(
            str(record.get("question", "")).strip(),
            str(record.get("instruction", "")).strip(),
        )
        instruction = instruction_from_record(
            str(record.get("instruction", "")).strip()
        )
        record_type = str(record.get("type", "")).strip().lower()

        docs = parse_docs_bundle(record.get("docs", ""))
        doc_payloads: List[Dict[str, Any]] = []
        for doc in docs:
            units = self._build_units(doc["content"])
            anchors = self._anchor_doc_with_llm(
                query=query, instruction=instruction, doc=doc, units=units
            )
            self._enrich_doc_anchors(
                anchors=anchors, doc_title=doc["doc_title"], record_type=record_type
            )

            citation_edges = []
            if anchors and "_doc_citation_edges" in anchors[0]:
                citation_edges = anchors[0].pop("_doc_citation_edges")

            doc_map = []
            for a in anchors:
                v5 = a.get("v5_metadata") or {}
                # Prefer rendered_summary for retrieval; keep original as backup.
                summary_for_map = a.get("rendered_summary") or a.get("summary", "")
                doc_map.append(
                    {
                        "anchor_id": a["anchor_id"],
                        "anchor_title": a["anchor_title"],
                        "summary": summary_for_map,
                        "original_summary": a.get("summary", ""),
                        "key_entities": a["key_entities"],
                        "semantic_role": v5.get("anchor_semantic_role", "unknown"),
                        "content_type": v5.get("anchor_content_type", "paragraph"),
                        "owner_name": v5.get("anchor_owner_name", ""),
                        "time_scope": v5.get("anchor_time_scope", ""),
                    }
                )

            doc_payloads.append(
                {
                    "doc_id": doc["doc_id"],
                    "doc_title": doc["doc_title"],
                    "unit_count": len(units),
                    "anchor_count": len(anchors),
                    "anchors": anchors,
                    "doc_map": doc_map,
                    "citation_edges": citation_edges,
                }
            )
            write_json(
                sample_dir / "anchors.partial.json",
                {
                    "record_id": record.get("id"),
                    "progress": {
                        "completed_docs": len(doc_payloads),
                        "total_docs": len(docs),
                    },
                    "docs": doc_payloads,
                },
            )

        payload = {
            "record_id": record.get("id"),
            "sample_id": safe_filename(
                f"{record.get('type','?')}_level{record.get('level','?')}_{record.get('id','?')}"
            ),
            "query": query,
            "docs": doc_payloads,
        }
        write_json(cache_path, payload)
        return payload
