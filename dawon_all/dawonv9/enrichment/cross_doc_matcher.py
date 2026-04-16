"""Deterministic cross-document citation matching (new in dawonv7).

v6 analysis showed the LLM Refiner almost always emits an empty
``relations.records`` list for paper-level citation tasks. Root cause: the
per-doc search facts contain references like "LLaMA" / "MS MARCO", but the
Refiner never gets an explicit *Documents roster* and never sees that one
provided document's title may appear inside another document's reference
list.

This module does that cross-check deterministically *before* the LLM runs, so
the Refiner can build on real signal instead of having to derive it from a
flat fact dump.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


_TITLE_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "the", "of", "in", "on", "for", "to", "with", "via",
    "by", "is", "at", "as", "from", "open", "large", "small", "language",
    "models", "model", "paper", "study", "survey",
}


def _normalize(text: str) -> str:
    return _TITLE_CLEAN_RE.sub(" ", (text or "").lower()).strip()


def _content_tokens(text: str) -> List[str]:
    return [t for t in _normalize(text).split() if t and t not in _STOPWORDS and len(t) > 2]


def _title_signature(title: str) -> Tuple[str, Set[str]]:
    norm = _normalize(title)
    return norm, set(_content_tokens(title))


def _iter_searchable_fields(item: Dict[str, Any]) -> Iterable[str]:
    for key in ("fact", "evidence_span", "normalized_value", "value"):
        val = item.get(key)
        if isinstance(val, str) and val:
            yield val
    for ent in item.get("entities") or []:
        if isinstance(ent, str) and ent:
            yield ent


def _haystack_for_doc(doc_result: Dict[str, Any]) -> str:
    parts: List[str] = []
    for item in doc_result.get("items", []) or []:
        for field in _iter_searchable_fields(item):
            parts.append(field)
    return _normalize(" \n ".join(parts))


def _score_match(title_norm: str, title_tokens: Set[str], haystack_norm: str) -> Tuple[float, str]:
    """Return (score in [0,1], kind) for (title -> haystack).

    dawonv9 additions
    -----------------
    * contains: one string is a substring of the other (e.g. a shorter
      variant title appears verbatim inside a longer reference string).
    * prefix_match: the first 2-3 significant tokens of the title all
      appear in the haystack — common when two papers share a model name
      prefix but differ in subtitle (e.g. "HuBERT: Self-Supervised …" vs
      "HuBERT: How much can a bad teacher …").
    * first_word_match: the single most distinctive token (≥5 chars) of
      the title appears in the haystack — weak but useful for acronyms
      (BERT, wav2vec, GPT, etc.).
    """
    if not title_norm or not haystack_norm:
        return 0.0, ""

    # Exact / contains
    if title_norm in haystack_norm or haystack_norm in title_norm:
        return 1.0 if title_norm == haystack_norm else 0.85, "contains"

    if not title_tokens:
        return 0.0, ""

    # Token overlap (original logic)
    present = sum(1 for tok in title_tokens if tok in haystack_norm)
    coverage = present / max(1, len(title_tokens))
    if coverage >= 0.8 and present >= 3:
        return coverage, "token_coverage"
    if coverage >= 0.6 and present >= 4:
        return coverage * 0.8, "partial_token_coverage"

    # Prefix match: check first 3 significant ordered tokens
    token_list = [t for t in title_norm.split() if t in title_tokens and len(t) >= 3]
    lead = token_list[:3]
    if len(lead) >= 2:
        lead_present = sum(1 for t in lead if t in haystack_norm)
        if lead_present >= len(lead):
            return 0.65, "prefix_match"

    # First-word / acronym match (e.g. "hubert", "wav2vec", "bert")
    long_lead = [t for t in token_list if len(t) >= 5][:1]
    if long_lead and long_lead[0] in haystack_norm:
        return 0.50, "first_word_match"

    return 0.0, ""


def detect_cross_doc_citations(
    *,
    docs: Sequence[Dict[str, Any]],
    doc_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Detect (cited_doc, citing_doc) pairs by matching titles into per-doc facts.

    Parameters
    ----------
    docs: anchors[].docs, each with doc_id + doc_title
    doc_results: per-doc SearchAgent output (has items with fact/entities/...)

    Returns
    -------
    {
      "roster": [{"doc_id": "DOC1", "doc_title": "...", "tokens": [...]}, ...],
      "matches": [{"cited_doc_id": "DOC1", "citing_doc_id": "DOC2",
                    "score": 0.92, "kind": "exact",
                    "evidence": "first 200-char excerpt"}],
      "adjacency": {"DOC1": {"cited_by": ["DOC2"], "cites": []}, ...},
    }
    """
    roster: List[Dict[str, Any]] = []
    sigs: Dict[str, Tuple[str, Set[str]]] = {}
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "").strip()
        title = str(doc.get("doc_title") or "").strip()
        if not doc_id or not title:
            continue
        norm, toks = _title_signature(title)
        sigs[doc_id] = (norm, toks)
        roster.append({"doc_id": doc_id, "doc_title": title, "tokens": sorted(toks)})

    haystacks: Dict[str, str] = {}
    raw_haystacks: Dict[str, str] = {}
    for res in doc_results:
        did = str(res.get("doc_id") or "").strip()
        if not did:
            continue
        raw = " \n ".join(
            field
            for item in (res.get("items") or [])
            for field in _iter_searchable_fields(item)
        )
        raw_haystacks[did] = raw
        haystacks[did] = _normalize(raw)

    matches: List[Dict[str, Any]] = []
    adjacency: Dict[str, Dict[str, List[str]]] = {
        did: {"cited_by": [], "cites": []} for did in sigs
    }

    for cited_id, (title_norm, tokens) in sigs.items():
        for citing_id, haystack in haystacks.items():
            if citing_id == cited_id:
                continue
            score, kind = _score_match(title_norm, tokens, haystack)
            if score <= 0:
                continue
            raw = raw_haystacks.get(citing_id, "")
            idx = raw.lower().find(title_norm[:30]) if title_norm else -1
            if idx < 0 and tokens:
                for tok in tokens:
                    idx = raw.lower().find(tok)
                    if idx >= 0:
                        break
            evidence = raw[max(0, idx - 40): idx + 200] if idx >= 0 else raw[:200]
            matches.append(
                {
                    "cited_doc_id": cited_id,
                    "citing_doc_id": citing_id,
                    "score": round(score, 3),
                    "kind": kind,
                    "evidence": evidence.strip(),
                }
            )
            if citing_id not in adjacency[cited_id]["cited_by"]:
                adjacency[cited_id]["cited_by"].append(citing_id)
            if cited_id not in adjacency.setdefault(citing_id, {"cited_by": [], "cites": []})["cites"]:
                adjacency[citing_id]["cites"].append(cited_id)

    matches.sort(key=lambda m: (-m["score"], m["cited_doc_id"], m["citing_doc_id"]))
    return {"roster": roster, "matches": matches, "adjacency": adjacency}


def format_for_refiner(cross_doc: Dict[str, Any]) -> str:
    """Render cross-doc findings into a compact prompt block."""
    roster_lines = [
        f"- {e['doc_id']}: {e['doc_title']}" for e in cross_doc.get("roster", [])
    ]
    match_lines = []
    for m in cross_doc.get("matches", [])[:20]:
        match_lines.append(
            f"- {m['citing_doc_id']} appears to cite {m['cited_doc_id']} "
            f"(score={m['score']}, kind={m['kind']}); "
            f"evidence: {m['evidence'][:160]}"
        )
    if not match_lines:
        match_lines.append("- (no deterministic cross-doc title matches detected)")
    return (
        "Documents roster:\n"
        + "\n".join(roster_lines)
        + "\n\nDeterministic cross-doc citation matches (verify before trusting):\n"
        + "\n".join(match_lines)
    )


def longest_citation_chain(adjacency: Dict[str, Dict[str, List[str]]]) -> List[str]:
    """Return the longest linear chain DOC_a -> DOC_b -> DOC_c where each
    cites the next, using the adjacency built by detect_cross_doc_citations.
    """
    nodes = list(adjacency.keys())
    best: List[str] = []
    for start in nodes:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            cites = adjacency.get(node, {}).get("cites", [])
            extended = False
            for nxt in cites:
                if nxt in path:
                    continue
                extended = True
                stack.append((nxt, path + [nxt]))
            if not extended and len(path) > len(best):
                best = path
    return best
