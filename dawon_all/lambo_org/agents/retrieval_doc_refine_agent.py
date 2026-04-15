"""RetrievalDocRefineAgent — cleaner baseline replacement for ``DocRefineAgent``.

What was REMOVED from ``lambo.agents.doc_refine_agent.DocRefineAgent``
---------------------------------------------------------------------
The original agent ran a think→search→info loop, calling the LLM each
round to decide which anchor (via its LLM-generated ``summary``) to open
next. That loop consumed the summary-based doc_map produced by the old
``AnchorAgent``. Since this baseline does not produce LLM summaries, the
loop has no useful signal to rank on.

What was REPLACED
-----------------
This agent simply selects the top-K chunks (pre-ranked by embedding
similarity inside :class:`RetrievalAnchorAgent`), concatenates their raw
text as evidence, and emits a per-document sheet in the same schema the
downstream :class:`GlobalComposer` expects.

Output schema
-------------
Per-doc sheet contains: ``doc_id``, ``doc_title``, ``scan_result``
(one of ``found`` / ``no_evidence``), ``evidence`` (concatenated chunks),
``rounds_used`` (always 1 — single retrieval pass), plus a structured
``retrieved_anchors`` list for traceability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from lambo.common import read_json, write_json


class RetrievalDocRefineAgent:
    def __init__(self, top_k_per_doc: int = 6, max_evidence_chars: int = 6000) -> None:
        self.top_k_per_doc = top_k_per_doc
        self.max_evidence_chars = max_evidence_chars
        # kept for signature compatibility with the original agent
        self.max_rounds = 1

    def run(
        self,
        *,
        question: str,
        instruction: str,
        record: Dict[str, Any],
        doc_payload: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
    ) -> Dict[str, Any]:
        doc_id = doc_payload["doc_id"]
        cache_path = sample_dir / f"doc_refine_{doc_id}.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        anchors = doc_payload.get("anchors", [])
        ranked = sorted(
            anchors,
            key=lambda a: (
                a.get("retrieval_rank", 10**9)
                if a.get("retrieval_rank", -1) >= 0
                else 10**9,
                a.get("order", 0),
            ),
        )
        top = [a for a in ranked if a.get("retrieval_rank", -1) >= 0][: self.top_k_per_doc]
        if not top:
            top = ranked[: self.top_k_per_doc]

        # Keep evidence in original document order for readability.
        top_in_order = sorted(top, key=lambda a: a.get("order", 0))

        pieces: List[str] = []
        used = 0
        retrieved_records: List[Dict[str, Any]] = []
        for anchor in top_in_order:
            block = (
                f"[{anchor['anchor_id']}] "
                f"(score={anchor.get('retrieval_score', 0.0):.4f})\n"
                f"{anchor.get('text', '').strip()}"
            )
            if used + len(block) > self.max_evidence_chars and pieces:
                break
            pieces.append(block)
            used += len(block)
            retrieved_records.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "retrieval_rank": anchor.get("retrieval_rank", -1),
                    "retrieval_score": anchor.get("retrieval_score", 0.0),
                    "preview": anchor.get("preview", ""),
                }
            )

        evidence = "\n\n".join(pieces).strip()
        sheet = {
            "doc_id": doc_id,
            "doc_title": doc_payload.get("doc_title", ""),
            "scan_result": "found" if evidence else "no_evidence",
            "evidence": evidence,
            "rounds_used": 1,
            "retrieved_anchors": retrieved_records,
        }
        write_json(cache_path, sheet)
        return sheet
