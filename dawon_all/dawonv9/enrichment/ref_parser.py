"""Reference section parser for academic paper anchors (dawonv9).

Extracts individual cited paper titles from reference list anchors so that
cross-document citation matching can work on explicit title strings rather
than raw token overlap against a long evidence blob.

Supports:
  IEEE / ICASSP style:
    [23] A. Author et al., "HuBERT: How much can a bad teacher..." ICASSP, 2021.
  NeurIPS / ICML style (unquoted, after period following authors):
    [3] Baevski et al. wav2vec 2.0: A framework... NeurIPS, 2020.
  ACL anthology style:
    Author (Year). Title of paper. In Proceedings...
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Reference-section detection
# ---------------------------------------------------------------------------

_BRACKET_REF_RE = re.compile(r"\[\d{1,3}\]")
_MIN_BRACKET_ENTRIES = 3


def is_reference_section(text: str) -> bool:
    """Return True if the anchor text looks like a bibliography section."""
    return len(_BRACKET_REF_RE.findall(text)) >= _MIN_BRACKET_ENTRIES


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------

# Pattern 1: "Title in double curly/smart/ASCII quotes"
_QUOTED_TITLE_RE = re.compile(
    r'["\u201c\u2018]'          # open quote (ASCII " or Unicode " or ')
    r'([^"\u201d\u2019]{10,220})'  # title body (10-220 chars)
    r'[",\u201d\u2019]',         # close quote or trailing comma
    re.DOTALL,
)

# Pattern 2: After [N] + authors (period-separated), unquoted title until next comma/period
# e.g.: [3] Baevski et al. wav2vec 2.0: A framework...
_UNQUOTED_ENTRY_RE = re.compile(
    r"\[\d{1,3}\]\s+"            # [N]
    r"(?:[A-Z][^.]+\.\s+)?"      # optional author block ending with period + space
    r"([A-Z\d][^,\n]{15,180})"  # title: starts capital/digit, 15-180 chars
    r"(?:,|\.|$)",
    re.MULTILINE,
)

# Pattern 3: arXiv style "Title. arXiv preprint arXiv:XXXX"
_ARXIV_TITLE_RE = re.compile(
    r'([A-Z][^.\n]{15,180})\.\s+(?:arXiv|CoRR)',
)


def _clean_title(raw: str) -> str:
    """Strip trailing punctuation and whitespace noise."""
    t = raw.strip().rstrip(".,;:\"'\u201d\u2019 ")
    # Remove "in Proceedings of..." suffix if it crept in
    t = re.split(r"\.\s+[Ii]n\s+", t)[0]
    t = re.split(r",\s+(?:in|In|proceedings|Proceedings)", t)[0]
    return t.strip()


def extract_cited_titles(anchor_text: str) -> List[str]:
    """Return a deduplicated list of paper titles found in the reference section.

    Tries quoted extraction first (high precision), then falls back to
    unquoted and arXiv patterns for entries without quotation marks.
    """
    seen: set = set()
    titles: List[str] = []

    def _add(raw: str) -> None:
        t = _clean_title(raw)
        key = t.lower()
        if len(t) >= 10 and key not in seen:
            seen.add(key)
            titles.append(t)

    # Pass 1: quoted titles (highest precision)
    for m in _QUOTED_TITLE_RE.finditer(anchor_text):
        _add(m.group(1))

    # Pass 2: unquoted entries for anything not caught by pass 1
    for m in _UNQUOTED_ENTRY_RE.finditer(anchor_text):
        candidate = m.group(1).strip()
        # Skip URLs, author-name fragments ("et al", "Author, Year"), short strings
        if re.search(r"https?://|www\.", candidate):
            continue
        if re.search(r"\bet\s+al\b|et al\.", candidate, re.IGNORECASE):
            continue
        if len(candidate) < 20:
            continue
        if candidate.lower() in seen:
            continue
        _add(candidate)

    # Pass 3: arXiv-style titles
    for m in _ARXIV_TITLE_RE.finditer(anchor_text):
        _add(m.group(1))

    return titles


# ---------------------------------------------------------------------------
# Convenience: scan all anchors in a doc and return (anchor_id, [titles])
# ---------------------------------------------------------------------------

def extract_ref_titles_from_doc(
    anchors: List[dict],
) -> List[Tuple[str, List[str]]]:
    """Scan all anchors in a document and return reference section findings.

    Returns list of (anchor_id, [cited_title, ...]) for every anchor that
    looks like a reference section.
    """
    results = []
    for anchor in anchors:
        text = anchor.get("text", "")
        if is_reference_section(text):
            titles = extract_cited_titles(text)
            if titles:
                results.append((anchor["anchor_id"], titles))
    return results
