"""Rule-based chunk metadata extraction for lambo_org_rbmetadata.

Motivation
----------
Plain chunking loses context: if a chunk cuts through the middle of a table,
a paper section, or a court judgment, the downstream LLM no longer knows
*which* company's row it is, *which* section of the paper it came from, or
*which* case it belongs to. This module attaches a small, deterministic,
rule-based metadata block to every chunk so that each chunk is interpretable
on its own — or together with other chunks from the same source.

Design constraints
------------------
* NO LLM calls. Everything here is regex / heuristics.
* Cheap enough to run on every chunk.
* Output is stable JSON-serializable dicts.

What we extract
---------------
Document-level (shared across all chunks of a doc):
  * owner_type     — company / paper / court_case / regulation / contract / other
  * owner_name     — issuing entity / author / court (best-effort)
  * doc_kind_hint  — table-heavy / paper-like / legal-like / mixed
  * table_headers  — list of column-header rows detected in the doc
  * section_index  — list of (char_offset, heading_text) for section headings
  * doc_case_id    — court case number (legal docs)

Chunk-level (per chunk, combining doc metadata + local analysis):
  * doc_id, doc_title, owner_type, owner_name   (copied from doc meta)
  * section_path    — nearest preceding heading for this chunk
  * content_type    — table / table_fragment / paragraph / list / figure / quote
  * table_header    — if content_type is table*, the header row(s) for its table
  * semantic_role   — definition / holding / case_fact / methodology / result / ...
  * time_scope      — "2023", "2023Q2", "2019-2021" style normalized period
  * unit_hints      — currency / percentage markers
  * case_id         — court case id if mentioned locally
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Document-level patterns
# ---------------------------------------------------------------------------

_OWNER_PATTERNS: List[Dict[str, Any]] = [
    {"regex": r"([\u4e00-\u9fff]+(?:法院|仲裁委))", "type": "court_case"},
    {"regex": r"判决文书|裁定书|裁判文书|判决书", "type": "court_case"},
    {"regex": r"([\u4e00-\u9fff]+(?:股份有限公司|集团|控股|公司))", "type": "company"},
    {"regex": r"证券简称[:：]\s*([^\s|　]+)", "type": "company"},
    {"regex": r"\b(arXiv|doi|proceedings|journal|Abstract)\b", "type": "paper", "flags": re.IGNORECASE},
    {"regex": r"([\u4e00-\u9fff]+(?:法|条例|规定|办法|准则|规则))$", "type": "regulation"},
    {"regex": r"(合同|协议|契约|agreement|contract)", "type": "contract", "flags": re.IGNORECASE},
]

# Court case id like "(2019)沪01民终12345号"
_CASE_ID_RE = re.compile(r"[(（]\s*\d{4}\s*[)）][\u4e00-\u9fff0-9]{2,}第?\s*\d+\s*号")

_HEADING_RE = re.compile(
    r"(?m)^\s*(?:"
    r"(?:\d+(?:\.\d+){0,3}\.?)\s+.{1,120}"  # "1." / "2.3.1 Introduction"
    r"|#{1,6}\s+.{1,120}"                    # markdown headings
    r"|第[一二三四五六七八九十百0-9]+[章节条]\s*.{0,80}"  # 第三章 ...
    r"|[IVXLC]+\.\s+.{1,120}"                 # Roman numeral sections
    r"|(?:Abstract|Introduction|Methods?|Methodology|Experiments?|Results?|Discussion|Conclusions?|References?)\s*$"
    r")\s*$"
)

_TIME_PATTERNS = [
    re.compile(r"(\d{4})\s*(?:年)?\s*(?:第?([一二三四1-4])季度|Q([1-4]))", re.IGNORECASE),
    re.compile(r"(\d{4})\s*[~\-–至到]\s*(\d{4})"),
    re.compile(r"(\d{4})\s*(?:年度|年|fiscal year)", re.IGNORECASE),
]
_QUARTER_MAP = {"一": "Q1", "二": "Q2", "三": "Q3", "四": "Q4",
                "1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}

_UNIT_PATTERNS = [
    re.compile(r"(百万|万|亿|千)\s*(?:元|美元|韩元|日元|港元|欧元|英镑)"),
    re.compile(r"单位[:：]\s*([\u4e00-\u9fff]+)"),
    re.compile(r"(元|美元|韩元|USD|CNY|KRW|EUR|GBP|JPY)", re.IGNORECASE),
    re.compile(r"(%|百分比|百分点|bps)"),
]

_ROLE_KEYWORDS: List[Tuple[str, str]] = [
    (r"(定义|definition|是指|means|refers to)", "definition"),
    (r"(本院认为|court holds|holding|认为.*应当)", "holding"),
    (r"(经审理查明|事实认定|facts found|案件事实)", "case_fact"),
    (r"(原告|被告|上诉|plaintiff|defendant|诉讼请求)", "legal_argument"),
    (r"(方法|methodology|approach|algorithm|architecture)", "methodology"),
    (r"(实验|experiment|evaluation|benchmark|ablation)", "result"),
    (r"(accuracy|precision|recall|f1\b|性能)", "result"),
    (r"(comparison|compared to|versus|vs\.?|相比)", "comparison"),
    (r"(reference|bibliography|参考文献|\[\d+\])", "citation_context"),
    (r"(evidence|证据|证明|表明|demonstrates)", "evidence"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_table_row(line: str) -> bool:
    """A table row has at least 3 pipe separators or 3 tab-separated columns."""
    return line.count("|") >= 3 or line.count("\t") >= 2


def _is_table_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and set(s) <= set("-|: \t") and "|" in s and s.count("-") >= 3


def _detect_table_spans(text: str) -> List[Dict[str, Any]]:
    """Scan doc text and return a list of table spans with their header rows.

    Each entry: {"start": char_offset, "end": char_offset, "header": str}
    """
    spans: List[Dict[str, Any]] = []
    lines = text.splitlines(keepends=True)
    offsets: List[int] = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)

    i = 0
    while i < len(lines):
        if _is_table_row(lines[i]):
            j = i
            while j < len(lines) and (_is_table_row(lines[j]) or _is_table_separator(lines[j]) or not lines[j].strip()):
                # break out on multiple blank lines
                if not lines[j].strip() and j + 1 < len(lines) and not lines[j + 1].strip():
                    break
                j += 1
            # Only treat as a table if the run has >= 2 table rows
            row_count = sum(1 for k in range(i, j) if _is_table_row(lines[k]))
            if row_count >= 2:
                header = lines[i].strip()
                start = offsets[i]
                end = offsets[j - 1] + len(lines[j - 1]) if j - 1 >= 0 else offsets[i]
                spans.append({"start": start, "end": end, "header": header})
            i = max(j, i + 1)
        else:
            i += 1
    return spans


def _detect_headings(text: str) -> List[Tuple[int, str]]:
    """Return list of (char_offset, heading_text) in the order they appear."""
    out: List[Tuple[int, str]] = []
    for m in _HEADING_RE.finditer(text):
        out.append((m.start(), m.group(0).strip()))
    return out


def _find_nearest_heading(headings: List[Tuple[int, str]], char_offset: int) -> str:
    """Binary search the nearest heading at or before char_offset."""
    if not headings:
        return ""
    lo, hi = 0, len(headings) - 1
    result = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        if headings[mid][0] <= char_offset:
            result = headings[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def _find_table_span(spans: List[Dict[str, Any]], char_offset: int) -> Optional[Dict[str, Any]]:
    for s in spans:
        if s["start"] <= char_offset < s["end"]:
            return s
    return None


def _classify_content_type(text: str, chunk_span: List[int], table_span: Optional[Dict[str, Any]]) -> str:
    if table_span is not None:
        # It's inside a table span. If the header row is literally contained in
        # the chunk text, it's a full 'table' chunk; else 'table_fragment'.
        header = (table_span or {}).get("header", "")
        if header and header in text:
            return "table"
        return "table_fragment"
    if re.search(r"(图|figure|fig\.)\s*\d+", text[:300], re.IGNORECASE):
        return "figure"
    if re.match(r"^\s*[-*•]\s+", text) or re.match(r"^\s*\d+[.)]\s+", text):
        return "list"
    if re.match(r"^\s*>", text):
        return "quote"
    return "paragraph"


def _classify_semantic_role(text: str) -> str:
    sample = text[:800]
    for pat, role in _ROLE_KEYWORDS:
        if re.search(pat, sample, re.IGNORECASE):
            return role
    return "unknown"


def _extract_time_scope(text: str) -> str:
    for pat in _TIME_PATTERNS:
        m = pat.search(text[:600])
        if not m:
            continue
        groups = m.groups()
        if len(groups) >= 3 and (groups[1] or groups[2]):
            return f"{groups[0]}{_QUARTER_MAP.get(groups[1] or groups[2] or '', '')}"
        if len(groups) == 2 and groups[1]:
            return f"{groups[0]}-{groups[1]}"
        return groups[0]
    return ""


def _extract_unit_hints(text: str) -> List[str]:
    hints: List[str] = []
    seen = set()
    for pat in _UNIT_PATTERNS:
        for m in pat.finditer(text[:1500]):
            hint = (m.group(0) or "").strip()
            if hint and hint not in seen:
                seen.add(hint)
                hints.append(hint)
    return hints[:5]


def _extract_case_id(text: str) -> str:
    m = _CASE_ID_RE.search(text)
    return m.group(0) if m else ""


def _extract_owner(doc_title: str, record_type: str) -> Tuple[str, str]:
    type_hint_map = {"financial": "company", "legal": "court_case", "paper": "paper"}
    default_type = type_hint_map.get(record_type, "other")
    for pat in _OWNER_PATTERNS:
        flags = pat.get("flags", 0)
        m = re.search(pat["regex"], doc_title, flags=flags)
        if m:
            name = next((g for g in m.groups() if g), "") if m.lastindex else ""
            return pat["type"], (name or re.sub(r"[《》「」]", "", doc_title).strip())
    return default_type, re.sub(r"[《》「」]", "", doc_title).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_doc_metadata(
    *,
    doc_id: str,
    doc_title: str,
    content: str,
    record_type: str = "",
) -> Dict[str, Any]:
    """Compute metadata shared by all chunks of a single document."""
    owner_type, owner_name = _extract_owner(doc_title, record_type)
    table_spans = _detect_table_spans(content)
    headings = _detect_headings(content)
    doc_case_id = _extract_case_id(content[:4000])

    # Simple doc_kind_hint
    total = max(1, len(content))
    table_chars = sum(s["end"] - s["start"] for s in table_spans)
    table_ratio = table_chars / total
    if table_ratio > 0.35:
        doc_kind_hint = "table_heavy"
    elif headings and owner_type == "paper":
        doc_kind_hint = "paper_like"
    elif owner_type == "court_case" or doc_case_id:
        doc_kind_hint = "legal_like"
    else:
        doc_kind_hint = "mixed"

    return {
        "doc_id": doc_id,
        "doc_title": doc_title,
        "owner_type": owner_type,
        "owner_name": owner_name,
        "doc_kind_hint": doc_kind_hint,
        "doc_case_id": doc_case_id,
        "table_spans": table_spans,         # list of {start,end,header}
        "headings": [{"offset": o, "text": t} for o, t in headings],
        "record_type": record_type,
    }


def build_chunk_metadata(
    *,
    chunk_text: str,
    char_span: List[int],
    doc_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute per-chunk metadata given doc-level context."""
    start = int(char_span[0]) if char_span else 0
    headings_tuples = [(h["offset"], h["text"]) for h in doc_meta.get("headings", [])]
    section_path = _find_nearest_heading(headings_tuples, start)
    table_span = _find_table_span(doc_meta.get("table_spans", []), start)
    content_type = _classify_content_type(chunk_text, char_span, table_span)
    semantic_role = _classify_semantic_role(chunk_text)
    time_scope = _extract_time_scope(chunk_text)
    unit_hints = _extract_unit_hints(chunk_text)
    local_case_id = _extract_case_id(chunk_text) or doc_meta.get("doc_case_id", "")
    table_header = (table_span or {}).get("header", "") if table_span else ""

    return {
        "doc_id": doc_meta["doc_id"],
        "doc_title": doc_meta["doc_title"],
        "owner_type": doc_meta["owner_type"],
        "owner_name": doc_meta["owner_name"],
        "doc_kind_hint": doc_meta["doc_kind_hint"],
        "section_path": section_path,
        "content_type": content_type,
        "table_header": table_header,
        "semantic_role": semantic_role,
        "time_scope": time_scope,
        "unit_hints": unit_hints,
        "case_id": local_case_id,
    }


def render_metadata_header(meta: Dict[str, Any]) -> str:
    """Compact, deterministic header string prepended to a chunk for the LLM.

    Only emit info that carries *new* signal beyond what's already visible in
    ``[DOC1_C003]``-style chunk ids and the doc_title: namely the chunk's
    section path, whether it's a table fragment (plus reattached header), and
    identifiers that ground the chunk to a larger entity (legal case id,
    currency/unit hints).

    Default-empty fields are omitted so the LLM isn't distracted by repetitive
    noise (owner_name == doc_title for papers, generic "paragraph" kinds, etc.).
    """
    parts: List[str] = []
    content_type = meta.get("content_type", "")
    if content_type in {"table", "table_fragment", "list", "figure", "quote"}:
        parts.append(f"kind={content_type}")
    if meta.get("section_path"):
        sp = meta["section_path"]
        if len(sp) > 80:
            sp = sp[:77] + "..."
        parts.append(f"section={sp}")
    semantic_role = meta.get("semantic_role", "")
    if semantic_role and semantic_role != "unknown":
        parts.append(f"role={semantic_role}")
    if meta.get("time_scope"):
        parts.append(f"time={meta['time_scope']}")
    if meta.get("case_id"):
        parts.append(f"case={meta['case_id']}")
    # Only name the owner when it actually adds new info (different from doc_title)
    owner_name = meta.get("owner_name", "")
    doc_title = meta.get("doc_title", "")
    if owner_name and owner_name != doc_title and len(owner_name) < 60:
        parts.append(f"owner={owner_name}")
    if meta.get("unit_hints"):
        parts.append(f"units={'/'.join(meta['unit_hints'])}")
    header = " | ".join(parts)
    if meta.get("table_header"):
        header += f"\ntable_header: {meta['table_header']}"
    return header
