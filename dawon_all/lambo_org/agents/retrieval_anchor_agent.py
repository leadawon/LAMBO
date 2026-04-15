"""RetrievalAnchorAgent — cleaner baseline replacement for ``AnchorAgent``.

What was REMOVED from ``lambo.agents.anchor_agent.AnchorAgent``
---------------------------------------------------------------
The original agent ran an LLM over each document to:
  * group units into semantic anchors,
  * generate an ``anchor_title``,
  * write a ``summary`` field,
  * extract ``key_entities``.

This was the core "anchor summary generation" contribution. This baseline
removes it entirely — no LLM call is made here.

What was REPLACED with retrieval
--------------------------------
* Each document is split into plain chunks (same paragraph-based chunker
  as ``_build_units`` in the original, copied locally so the package is
  standalone).
* Each chunk is embedded with sentence-transformers.
* The query (question + instruction) is embedded once per record.
* Chunks are ranked by cosine similarity against the query.
* Each chunk is exposed as a "pseudo-anchor" so that downstream modules
  (``DocRefineAgent`` / ``GlobalComposer`` / ``Generator``) continue to
  work with minimal change.

Anchor format compatibility
---------------------------
The emitted ``anchors`` list mirrors the original schema, with these
differences (clearly marked so no caller misreads them as LLM output):
  * ``anchor_title`` is a short preview-derived label, NOT LLM-generated.
  * ``summary`` is just the truncated raw chunk text, NOT LLM-generated.
  * ``key_entities`` is always an empty list — we do not extract entities.
  * ``retrieval_rank`` (new) is the chunk's rank after cosine similarity.
  * ``retrieval_score`` (new) is the cosine similarity against the query.

The ``anchors`` list is sorted in ORIGINAL DOCUMENT ORDER (to match the
downstream assumption), but each anchor carries its retrieval rank so the
doc-refine step can simply pick the top-K without a think/search loop.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from lambo.common import (
    compact_text,
    current_query_from_record,
    ensure_dir,
    instruction_from_record,
    parse_docs_bundle,
    read_json,
    safe_filename,
    write_json,
)

from ..embeddings import EmbeddingBackend, cosine_rank


def _build_units(document_text: str, max_units_per_doc: int = 220) -> List[Dict[str, Any]]:
    """Paragraph-based chunker copied from ``lambo.agents.anchor_agent``.

    Copied here intentionally to keep ``lambo_org`` importable without
    pulling in the LLM-driven AnchorAgent.
    """
    text = (document_text or "").replace("\r\n", "\n")
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    if len(raw_blocks) <= 1:
        raw_blocks = [line.strip() for line in text.splitlines() if line.strip()]

    blocks: List[str] = []
    for block in raw_blocks:
        if len(block) > 2400 and block.count("\n") >= 2:
            blocks.extend(ln.strip() for ln in block.splitlines() if ln.strip())
        else:
            blocks.append(block)

    merged: List[str] = []
    buffer = ""
    for block in blocks:
        if len(block) < 80 and buffer:
            buffer = buffer + "\n" + block
            if len(buffer) >= 160:
                merged.append(buffer)
                buffer = ""
        else:
            if buffer:
                merged.append(buffer)
            buffer = block
    if buffer:
        merged.append(buffer)

    if len(merged) > max_units_per_doc:
        stride = max(1, len(merged) // max_units_per_doc + 1)
        packed: List[str] = []
        group: List[str] = []
        for block in merged:
            group.append(block)
            if len(group) >= stride:
                packed.append("\n".join(group))
                group = []
        if group:
            packed.append("\n".join(group))
        merged = packed

    units: List[Dict[str, Any]] = []
    cursor = 0
    for index, block in enumerate(merged, start=1):
        start = text.find(block[:60], cursor) if block else cursor
        if start < 0:
            start = cursor
        end = start + len(block)
        cursor = max(cursor, end)
        units.append(
            {
                "unit_id": f"U{index:03d}",
                "order": index,
                "char_span": [start, end],
                "text": block,
                "preview": compact_text(block, limit=220),
            }
        )
    return units


class RetrievalAnchorAgent:
    """Embedding-similarity retrieval baseline. No LLM calls here."""

    def __init__(
        self,
        embedder: EmbeddingBackend,
        max_units_per_doc: int = 220,
        top_k_per_doc: int = 6,
        top_k_global: int = 24,
        retrieval_scope: str = "per_document",
    ) -> None:
        self.embedder = embedder
        self.max_units_per_doc = max_units_per_doc
        self.top_k_per_doc = top_k_per_doc
        self.top_k_global = top_k_global
        if retrieval_scope not in {"per_document", "global"}:
            raise ValueError(f"Invalid retrieval_scope: {retrieval_scope}")
        self.retrieval_scope = retrieval_scope

    @staticmethod
    def _pseudo_anchor(
        doc_id: str,
        doc_title: str,
        chunk_index: int,
        unit: Dict[str, Any],
    ) -> Dict[str, Any]:
        # IMPORTANT: anchor_title and summary are derived from raw text —
        # they are NOT LLM summaries. Key_entities is intentionally empty.
        text = unit["text"]
        return {
            "anchor_id": f"{doc_id}_C{chunk_index:03d}",
            "doc_id": doc_id,
            "doc_title": doc_title,
            "order": chunk_index,
            "anchor_title": compact_text(text.split("\n", 1)[0], limit=50)
            or f"Chunk {chunk_index}",
            "summary": compact_text(text, limit=240),
            "key_entities": [],
            "packet_span": f"{unit['unit_id']}..{unit['unit_id']}",
            "char_span": list(unit["char_span"]),
            "unit_ids": [unit["unit_id"]],
            "text": text,
            "preview": unit["preview"],
            "prev_anchor_id": "",
            "next_anchor_id": "",
            "retrieval_rank": -1,
            "retrieval_score": 0.0,
        }

    def _rank_chunks(
        self,
        query: str,
        doc_payloads: List[Dict[str, Any]],
    ) -> None:
        """Embed the query + all chunks once, then assign rank/score in place."""
        all_texts: List[str] = []
        owners: List[int] = []  # index into doc_payloads
        for di, doc in enumerate(doc_payloads):
            for anchor in doc["anchors"]:
                all_texts.append(anchor["text"])
                owners.append(di)

        if not all_texts:
            return

        chunk_matrix = self.embedder.encode(all_texts)
        query_vec = self.embedder.encode([query])[0]

        if self.retrieval_scope == "global":
            # Build (doc_index, offset_in_doc) for every flat chunk.
            flat_to_local: List[tuple] = []
            for di, doc in enumerate(doc_payloads):
                for offset in range(len(doc["anchors"])):
                    flat_to_local.append((di, offset))

            scores = (chunk_matrix @ query_vec).tolist()
            for flat_idx, (di, offset) in enumerate(flat_to_local):
                anchor = doc_payloads[di]["anchors"][offset]
                anchor["retrieval_score"] = float(scores[flat_idx])
                anchor["retrieval_rank"] = -1
                anchor["retrieved"] = False

            order = cosine_rank(query_vec, chunk_matrix)
            for rank_i, flat_idx in enumerate(order):
                di, offset = flat_to_local[flat_idx]
                anchor = doc_payloads[di]["anchors"][offset]
                anchor["retrieval_rank"] = rank_i
                if rank_i < self.top_k_global:
                    anchor["retrieved"] = True
        else:
            # per-document: rank chunks within each doc; top_k_per_doc flagged
            scores = chunk_matrix @ query_vec
            cursor = 0
            for doc in doc_payloads:
                n = len(doc["anchors"])
                doc_scores = scores[cursor : cursor + n]
                cursor += n
                order = list(sorted(range(n), key=lambda i: -doc_scores[i]))
                for rank_i, idx in enumerate(order):
                    doc["anchors"][idx]["retrieval_rank"] = rank_i
                    doc["anchors"][idx]["retrieval_score"] = float(doc_scores[idx])
                    doc["anchors"][idx]["retrieved"] = rank_i < self.top_k_per_doc

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
        instruction = instruction_from_record(str(record.get("instruction", "")).strip())
        retrieval_query = f"{query}\n{instruction}".strip()

        docs = parse_docs_bundle(record.get("docs", ""))
        doc_payloads: List[Dict[str, Any]] = []
        for doc in docs:
            units = _build_units(doc["content"], max_units_per_doc=self.max_units_per_doc)
            anchors = [
                self._pseudo_anchor(doc["doc_id"], doc["doc_title"], i, unit)
                for i, unit in enumerate(units, start=1)
            ]
            for i, anchor in enumerate(anchors):
                if i > 0:
                    anchor["prev_anchor_id"] = anchors[i - 1]["anchor_id"]
                if i < len(anchors) - 1:
                    anchor["next_anchor_id"] = anchors[i + 1]["anchor_id"]
            doc_payloads.append(
                {
                    "doc_id": doc["doc_id"],
                    "doc_title": doc["doc_title"],
                    "unit_count": len(units),
                    "anchor_count": len(anchors),
                    "anchors": anchors,
                    "doc_map": [
                        {
                            "anchor_id": a["anchor_id"],
                            "anchor_title": a["anchor_title"],
                            # NOTE: summary below is truncated raw text — not
                            # an LLM summary. Kept for interface compatibility.
                            "summary": a["summary"],
                            "key_entities": a["key_entities"],
                        }
                        for a in anchors
                    ],
                }
            )

        # Single-shot embedding rank across the whole record's chunks.
        self._rank_chunks(retrieval_query, doc_payloads)

        payload = {
            "record_id": record.get("id"),
            "sample_id": safe_filename(
                f"{record.get('type','?')}_level{record.get('level','?')}_{record.get('id','?')}"
            ),
            "query": query,
            "retrieval": {
                "scope": self.retrieval_scope,
                "top_k_per_doc": self.top_k_per_doc,
                "top_k_global": self.top_k_global,
                "embedding_model": self.embedder.model_name,
            },
            "docs": doc_payloads,
        }
        write_json(cache_path, payload)
        return payload
