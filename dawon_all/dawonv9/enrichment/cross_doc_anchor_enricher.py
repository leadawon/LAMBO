"""Cross-document anchor enrichment — Phase 2 (dawonv9).

After AnchorAgent has processed ALL documents in a sample, this module runs
a deterministic cross-document citation search and injects hints directly
into the anchor records BEFORE DocRefineAgent runs.

Why this matters
----------------
The original cross_doc_matcher runs AFTER DocRefineAgent on its evidence
output. If DocRefineAgent returned no_evidence (because the LLM gave up on
title mismatch), there is nothing to match. This module flips the order:
  1. Extract cited paper titles from every reference-section anchor.
  2. Fuzzy-match them against every other document's title in the sample.
  3. Tag matching anchors with ``cross_doc_cites`` so DocRefineAgent sees
     "★ cites DOC1 (HuBERT)" in the map and is guided to open that anchor.

Only activated for record_type == "paper".
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from .ref_parser import extract_ref_titles_from_doc, is_reference_section


# ---------------------------------------------------------------------------
# Title normalisation & matching (mirrors cross_doc_matcher logic but used
# here against extracted titles rather than evidence blobs)
# ---------------------------------------------------------------------------

_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_STOPWORDS: Set[str] = {
    "a", "an", "and", "the", "of", "in", "on", "for", "to", "with", "via",
    "by", "is", "at", "as", "from", "open", "large", "small", "language",
    "models", "model", "paper", "study", "survey", "based", "using", "pre",
    "training", "learning", "deep", "neural", "network", "approach",
}

_MIN_SCORE = 0.50          # minimum score to inject a hint
_EXACT_SCORE = 1.00
_TOKEN_HIGH_SCORE = 0.90   # ≥80% token overlap, ≥3 tokens
_TOKEN_MID_SCORE = 0.75    # ≥60% overlap, ≥4 tokens
_PREFIX_SCORE = 0.65       # first distinctive word match
_CONTAIN_SCORE = 0.80      # one title is a substring of the other


def _norm(text: str) -> str:
    return _CLEAN_RE.sub(" ", (text or "").lower()).strip()


def _content_tokens(text: str) -> List[str]:
    return [t for t in _norm(text).split() if t and t not in _STOPWORDS and len(t) > 2]


def _sig(title: str) -> Tuple[str, List[str]]:
    """Return (normalised_title, content_token_list)."""
    n = _norm(title)
    return n, _content_tokens(title)


def _score(doc_norm: str, doc_tokens: List[str], cited_norm: str) -> Tuple[float, str]:
    """Score how well ``doc_norm`` (a document title) matches ``cited_norm``
    (an extracted reference title from another document's bibliography).
    """
    if not doc_norm or not cited_norm:
        return 0.0, ""

    # Exact substring match
    if doc_norm in cited_norm or cited_norm in doc_norm:
        return _EXACT_SCORE if doc_norm == cited_norm else _CONTAIN_SCORE, "contains"

    if not doc_tokens:
        return 0.0, ""

    # Token overlap
    present = sum(1 for t in doc_tokens if t in cited_norm)
    coverage = present / len(doc_tokens)

    if coverage >= 0.8 and present >= 3:
        return _TOKEN_HIGH_SCORE * coverage, "token_coverage"
    if coverage >= 0.6 and present >= 4:
        return _TOKEN_MID_SCORE * coverage, "partial_token_coverage"

    # Prefix / first-word match
    # If the most distinctive leading token of the doc title appears in the
    # cited title, it's likely a variant of the same paper.
    lead_tokens = [t for t in doc_tokens if len(t) >= 4][:3]
    if lead_tokens:
        lead_present = sum(1 for t in lead_tokens if t in cited_norm)
        # All leading tokens found → strong prefix signal
        if lead_present == len(lead_tokens) and len(lead_tokens) >= 2:
            return _PREFIX_SCORE, "prefix_match"
        # The FIRST distinctive token (acronym / model name like "hubert",
        # "wav2vec") must itself appear in cited_norm — not just any token.
        if lead_tokens[0] in cited_norm and len(lead_tokens[0]) >= 5:
            return 0.55, "first_word_match"

    return 0.0, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_with_cross_doc_citations(
    doc_payloads: List[Dict[str, Any]],
    record_type: str,
) -> None:
    """Inject ``cross_doc_cites`` hints into reference-section anchors.

    Modifies *doc_payloads* in-place.  Does nothing for non-paper records.
    """
    if (record_type or "").lower() != "paper":
        return

    # Build roster: doc_id → (title, norm, tokens)
    roster: Dict[str, Tuple[str, str, List[str]]] = {}
    for dp in doc_payloads:
        doc_id = dp.get("doc_id", "")
        title = dp.get("doc_title", "")
        if doc_id and title:
            n, toks = _sig(title)
            roster[doc_id] = (title, n, toks)

    if len(roster) < 2:
        return

    # For each document, scan its anchors for reference sections
    for dp in doc_payloads:
        doc_id = dp.get("doc_id", "")
        anchors = dp.get("anchors", [])

        ref_findings = extract_ref_titles_from_doc(anchors)
        if not ref_findings:
            continue

        # Build a flat set of all cited titles from this doc's references
        anchor_titles_map: Dict[str, List[str]] = {}
        for anchor_id, cited_titles in ref_findings:
            anchor_titles_map[anchor_id] = cited_titles

        # For each reference-section anchor, match cited titles against
        # other documents in the sample
        anchor_by_id = {a["anchor_id"]: a for a in anchors}

        for anchor_id, cited_titles in anchor_titles_map.items():
            if anchor_id not in anchor_by_id:
                continue

            cross_cites: List[Dict[str, Any]] = []
            for other_id, (other_title, other_norm, other_tokens) in roster.items():
                if other_id == doc_id:
                    continue

                best_score = 0.0
                best_kind = ""
                best_via = ""

                for cited in cited_titles:
                    cited_n, _ = _sig(cited)
                    s, k = _score(other_norm, other_tokens, cited_n)
                    if s > best_score:
                        best_score = s
                        best_kind = k
                        best_via = cited

                if best_score >= _MIN_SCORE:
                    cross_cites.append({
                        "doc_id": other_id,
                        "doc_title": other_title,
                        "score": round(best_score, 3),
                        "kind": best_kind,
                        "matched_via": best_via,
                    })

            if cross_cites:
                cross_cites.sort(key=lambda x: -x["score"])
                anchor_by_id[anchor_id]["cross_doc_cites"] = cross_cites

                # Also tag the anchor's v5_metadata so the rendered summary
                # and doc_map formatter can surface it
                v5 = anchor_by_id[anchor_id].get("v5_metadata") or {}
                cited_ids = ", ".join(c["doc_id"] for c in cross_cites)
                v5["cross_doc_cites_summary"] = f"cites {cited_ids}"
                anchor_by_id[anchor_id]["v5_metadata"] = v5
