"""Build a lightweight structured-reading view for Loong set-1 records.

The goal of this script is not to answer with an LLM. It shows the
deterministic front half of a RAG-style reader:

1. split the bundled `docs` field into documents,
2. split each document into sections and chunks,
3. score documents/sections with domain-specific signals,
4. emit compact evidence snippets that a downstream model should read first.

It intentionally uses only the Python standard library so it can run inside a
plain venv, a docker container, or the repository's local Python.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "reference" / "Loong" / "data" / "loong_process.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "loong_structured_reading_demo"

TITLE_START = "<标题起始符>"
TITLE_END = "<标题终止符>"
DOC_END = "<doc终止符>"

IMPORTANT_PAPER_HEADINGS = (
    "abstract",
    "introduction",
    "related work",
    "method",
    "experiment",
    "conclusion",
    "reference",
    "bibliography",
)

FINANCIAL_SECTION_RE = re.compile(
    r"^\s*(part\s+[ivx]+\.?.*|item\s+\d+[a-z]?\.?.*|"
    r"(?:condensed\s+)?(?:consolidated\s+)?(?:balance sheets?|statements? of .+|"
    r"notes? to .+|financial statements?).*)\s*$",
    re.IGNORECASE,
)

MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.{1,220}?)\s*$")

LEGAL_MARKERS = (
    "本院认为",
    "本院查明",
    "经审查查明",
    "审理查明",
    "原审法院查明",
    "申请复议称",
    "复议申请人称",
    "赔偿请求人",
    "申请执行人",
    "被执行人",
    "裁定如下",
    "判决如下",
    "决定如下",
)

LEGAL_LABEL_TERMS = {
    "赔偿案件": ("赔偿", "国家赔偿", "司法赔偿", "无罪逮捕赔偿", "赔偿请求人"),
    "民事案件": ("民事", "合同纠纷", "申请再审", "民事裁定", "民事判决"),
    "执行案件": ("执行", "执行裁定", "执行审查", "执行复议", "申请执行人", "被执行人"),
}

EN_STOPWORDS = {
    "about",
    "above",
    "according",
    "after",
    "again",
    "against",
    "answer",
    "based",
    "before",
    "being",
    "between",
    "carefully",
    "company",
    "content",
    "could",
    "determine",
    "document",
    "documents",
    "does",
    "each",
    "following",
    "format",
    "found",
    "from",
    "given",
    "have",
    "ignore",
    "into",
    "only",
    "other",
    "papers",
    "part",
    "please",
    "provided",
    "question",
    "read",
    "review",
    "section",
    "solely",
    "statements",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "those",
    "type",
    "using",
    "what",
    "when",
    "which",
    "will",
    "with",
    "your",
    "per",
    "inc",
    "corp",
    "corporation",
    "company",
    "companies",
}

PAPER_GENERIC_TERMS = {
    "analysis",
    "approach",
    "based",
    "benchmark",
    "efficient",
    "effort",
    "evaluation",
    "framework",
    "language",
    "learning",
    "method",
    "model",
    "models",
    "neural",
    "open",
    "paper",
    "pre-training",
    "pretraining",
    "retrieval",
    "system",
    "systems",
    "towards",
    "using",
}

FINANCIAL_GENERIC_TERMS = {
    "basic",
    "company",
    "enterprise",
    "general",
    "inc",
    "share",
    "ventures",
}


@dataclass
class Section:
    section_id: str
    heading: str
    level: int
    text: str
    start_char: int

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class StructuredDoc:
    doc_id: str
    bundle_title: str
    source_name: str
    content: str
    sections: List[Section] = field(default_factory=list)

    @property
    def display_title(self) -> str:
        if self.source_name and self.source_name != self.bundle_title:
            return f"{self.source_name} / {self.bundle_title}"
        return self.source_name or self.bundle_title

    @property
    def char_count(self) -> int:
        return len(self.content)


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def compact(text: Any, limit: int = 260) -> str:
    value = normalize_ws(text)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_docs_bundle(raw_docs: str, source_names: Sequence[str]) -> List[StructuredDoc]:
    docs: List[StructuredDoc] = []
    parts = str(raw_docs or "").split(TITLE_START)
    for index, part in enumerate(parts[1:], start=1):
        if TITLE_END not in part:
            continue
        title, content = part.split(TITLE_END, 1)
        body = content.replace(DOC_END, "").strip()
        source_name = source_names[index - 1] if index - 1 < len(source_names) else ""
        docs.append(
            StructuredDoc(
                doc_id=f"D{index}",
                bundle_title=normalize_ws(title),
                source_name=normalize_ws(source_name),
                content=body,
            )
        )
    return docs


def iter_line_spans(text: str) -> Iterable[Tuple[int, str]]:
    offset = 0
    for line in text.splitlines(keepends=True):
        clean = line.rstrip("\r\n")
        yield offset, clean
        offset += len(line)


def split_markdown_or_financial_sections(text: str, doc_id: str, doc_type: str) -> List[Section]:
    hits: List[Tuple[int, int, str]] = []
    for offset, line in iter_line_spans(text):
        heading = ""
        level = 2
        md = MARKDOWN_HEADING_RE.match(line)
        if md:
            level = len(md.group(1))
            heading = normalize_ws(md.group(2))
        elif doc_type == "financial" and FINANCIAL_SECTION_RE.match(line):
            heading = normalize_ws(line)
            level = 2 if heading.lower().startswith(("part", "item")) else 3
        if heading:
            hits.append((offset, level, heading))

    if not hits:
        return [Section(f"{doc_id}:S1", "front matter", 1, text.strip(), 0)] if text.strip() else []

    sections: List[Section] = []
    if hits[0][0] > 0:
        front = text[: hits[0][0]].strip()
        if front:
            sections.append(Section(f"{doc_id}:S{len(sections) + 1}", "front matter", 1, front, 0))

    for idx, (start, level, heading) in enumerate(hits):
        end = hits[idx + 1][0] if idx + 1 < len(hits) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append(Section(f"{doc_id}:S{len(sections) + 1}", heading, level, body, start))
    return sections


def split_legal_sections(text: str, doc_id: str) -> List[Section]:
    marker_hits: List[Tuple[int, str]] = []
    for marker in LEGAL_MARKERS:
        for match in re.finditer(re.escape(marker), text):
            marker_hits.append((match.start(), marker))
    marker_hits.sort(key=lambda item: item[0])

    if not marker_hits:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) <= 1:
            return [Section(f"{doc_id}:S1", "全文", 1, text.strip(), 0)] if text.strip() else []
        sections: List[Section] = []
        cursor = 0
        for paragraph in paragraphs:
            start = text.find(paragraph, cursor)
            cursor = start + len(paragraph)
            sections.append(
                Section(
                    f"{doc_id}:S{len(sections) + 1}",
                    "段落",
                    2,
                    paragraph,
                    max(start, 0),
                )
            )
        return sections

    sections = []
    if marker_hits[0][0] > 0:
        front = text[: marker_hits[0][0]].strip()
        if front:
            sections.append(Section(f"{doc_id}:S1", "首部", 1, front, 0))

    for idx, (start, marker) in enumerate(marker_hits):
        end = marker_hits[idx + 1][0] if idx + 1 < len(marker_hits) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append(Section(f"{doc_id}:S{len(sections) + 1}", marker, 2, body, start))
    return sections


def split_sections(doc: StructuredDoc, doc_type: str) -> List[Section]:
    if doc_type == "legal":
        return split_legal_sections(doc.content, doc.doc_id)
    return split_markdown_or_financial_sections(doc.content, doc.doc_id, doc_type)


def sentenceish_split(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；;.!?])\s+|\n{2,}", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, target_chars: int = 1400) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= target_chars:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for part in sentenceish_split(text):
        if len(part) > target_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            chunks.extend(part[i : i + target_chars].strip() for i in range(0, len(part), target_chars))
            continue
        if current and current_len + len(part) + 1 > target_chars:
            chunks.append(" ".join(current).strip())
            current = [part]
            current_len = len(part)
        else:
            current.append(part)
            current_len += len(part) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def english_tokens(text: str) -> List[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_.&/-]{2,}|\$?-?\d+(?:\.\d+)?%?", str(text or ""))
    tokens: List[str] = []
    seen = set()
    for token in raw:
        value = token.strip(".,;:()[]{}").casefold()
        if len(value) < 3 or value in EN_STOPWORDS:
            continue
        if value not in seen:
            seen.add(value)
            tokens.append(value)
    return tokens


def quoted_terms(text: str) -> List[str]:
    matches = re.findall(r"[\"'“”‘’]([^\"'“”‘’]{1,100})[\"'“”‘’]", str(text or ""))
    return [normalize_ws(match) for match in matches if normalize_ws(match)]


def title_terms(title: str) -> List[str]:
    value = normalize_ws(title).strip("# ")
    if not value:
        return []
    pieces = [value]
    no_subtitle = re.split(r"[:：-]", value, maxsplit=1)[0].strip()
    if no_subtitle and no_subtitle != value:
        pieces.append(no_subtitle)
    for token in english_tokens(value):
        if token in PAPER_GENERIC_TERMS:
            continue
        if len(token) > 5 or re.search(r"[A-Z][a-z]+[A-Z]|[A-Z]{2,}", token):
            pieces.append(token)
    return unique_terms(pieces)


def unique_terms(terms: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for term in terms:
        cleaned = normalize_ws(term).strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def lower_count(text: str, term: str) -> int:
    if not term:
        return 0
    return text.casefold().count(term.casefold())


def literal_count(text: str, term: str) -> int:
    if not term:
        return 0
    if re.search(r"[A-Za-z]", term):
        return lower_count(text, term)
    return text.count(term)


def extract_snippets(text: str, terms: Sequence[str], window: int = 260, limit: int = 4) -> List[Dict[str, Any]]:
    snippets: List[Dict[str, Any]] = []
    seen_ranges = set()
    haystack_lower = text.casefold()
    for term in terms:
        if not term:
            continue
        needle = term.casefold()
        start = haystack_lower.find(needle) if re.search(r"[A-Za-z]", term) else text.find(term)
        while start != -1 and len(snippets) < limit:
            left = max(0, start - window)
            right = min(len(text), start + len(term) + window)
            range_key = (left // 80, right // 80)
            if range_key not in seen_ranges:
                seen_ranges.add(range_key)
                snippet_text = compact(text[left:right], window * 2 + 120)
                snippets.append({"term": term, "snippet": snippet_text})
            next_from = start + max(1, len(term))
            start = haystack_lower.find(needle, next_from) if re.search(r"[A-Za-z]", term) else text.find(term, next_from)
    return snippets


def section_inventory(doc: StructuredDoc, max_items: int = 8) -> List[Dict[str, Any]]:
    inventory = []
    for section in doc.sections[:max_items]:
        inventory.append(
            {
                "section_id": section.section_id,
                "heading": compact(section.heading, 80),
                "chars": section.char_count,
            }
        )
    return inventory


def infer_financial_terms(question: str) -> List[str]:
    q = normalize_ws(question)
    terms = quoted_terms(q)
    lower_q = q.casefold()
    matched_phrase = False
    phrase_map = {
        "basic earnings per share": ("basic earnings per share", "earnings per share", "basic", "per share"),
        "diluted earnings per share": ("diluted earnings per share", "earnings per share", "diluted", "per share"),
        "net income": ("net income", "net loss"),
        "revenue": ("revenue", "revenues", "sales"),
        "cash": ("cash and cash equivalents", "cash"),
        "assets": ("total assets", "assets"),
        "liabilities": ("total liabilities", "liabilities"),
        "stockholders": ("stockholders' equity", "shareholders' equity", "equity"),
    }
    for phrase, expansions in phrase_map.items():
        if phrase in lower_q:
            matched_phrase = True
            terms.extend(expansions)
    if not matched_phrase:
        terms.extend(token for token in english_tokens(q) if token not in FINANCIAL_GENERIC_TERMS)
    return unique_terms(terms)


def infer_legal_labels(question: str) -> List[str]:
    labels = []
    labels.extend(term for term in quoted_terms(question) if "案件" in term)
    for label in LEGAL_LABEL_TERMS:
        if label in question:
            labels.append(label)
    return unique_terms(labels)


def legal_label_scores(text: str) -> Dict[str, int]:
    scores = {}
    for label, terms in LEGAL_LABEL_TERMS.items():
        scores[label] = sum(text.count(term) for term in terms)
    return scores


def guess_legal_label(doc: StructuredDoc) -> Dict[str, Any]:
    # Titles are high signal in this dataset, so weight them more than body text.
    title_blob = f"{doc.source_name} {doc.bundle_title}"
    body_blob = doc.content[:2500]
    scores = defaultdict(int)
    for label, terms in LEGAL_LABEL_TERMS.items():
        for term in terms:
            scores[label] += 4 * title_blob.count(term)
            scores[label] += body_blob.count(term)
    if not scores:
        return {"label": "", "scores": {}}
    label, score = max(scores.items(), key=lambda item: item[1])
    return {"label": label if score > 0 else "", "scores": dict(sorted(scores.items()))}


def score_section(section: Section, terms: Sequence[str], doc_type: str) -> Tuple[float, List[str]]:
    text = f"{section.heading}\n{section.text}"
    reasons: List[str] = []
    score = 0.0
    for term in terms:
        count = literal_count(text, term)
        if count:
            weight = 2.0 if len(term) > 12 else 1.0
            score += min(8.0, count * weight)
            reasons.append(f"term:{term}x{count}")

    heading_l = section.heading.casefold()
    if doc_type == "paper":
        for heading in IMPORTANT_PAPER_HEADINGS:
            if heading in heading_l:
                score += 2.5
                reasons.append(f"paper-heading:{heading}")
                break
    elif doc_type == "financial":
        if any(key in heading_l for key in ("statement", "balance", "operations", "income", "note", "financial")):
            score += 3.0
            reasons.append("financial-table-section")
        if re.search(r"\$?\(?-?\d[\d,]*(?:\.\d+)?\)?", section.text):
            score += 1.0
            reasons.append("numeric-values")
    elif doc_type == "legal":
        if section.heading in LEGAL_MARKERS:
            score += 2.5
            reasons.append(f"legal-marker:{section.heading}")
    return score, reasons[:8]


def paper_reading_plan(record: Dict[str, Any], docs: Sequence[StructuredDoc]) -> Dict[str, Any]:
    question = normalize_ws(record.get("question", ""))
    provided_titles = [doc.source_name or doc.bundle_title for doc in docs]
    target_terms = title_terms(question)
    cross_doc_terms: List[str] = []
    for title in provided_titles:
        cross_doc_terms.extend(title_terms(title)[:3])
    terms = unique_terms(target_terms + cross_doc_terms)

    ranked_docs = []
    for doc in docs:
        title_blob = f"{doc.source_name} {doc.bundle_title}"
        text_blob = f"{title_blob}\n{doc.content[:10000]}"
        score = 0.0
        reasons = []
        if question and lower_count(title_blob, question):
            score += 20
            reasons.append("title-is-target-paper")
        target_mentions = sum(literal_count(doc.content, term) for term in target_terms)
        if target_mentions:
            score += min(20, target_mentions * 4)
            reasons.append(f"mentions-target:{target_mentions}")
        other_mentions = 0
        for title in provided_titles:
            if title != doc.source_name:
                other_mentions += literal_count(doc.content, title)
        if other_mentions:
            score += min(14, other_mentions * 3)
            reasons.append(f"mentions-provided-papers:{other_mentions}")
        if any(lower_count(text_blob, heading) for heading in ("references", "bibliography", "related work")):
            score += 3
            reasons.append("has-citation-sections")
        ranked_docs.append({"doc_id": doc.doc_id, "title": doc.display_title, "score": round(score, 2), "reasons": reasons})

    return {
        "strategy": [
            "Identify the target paper by exact/short title.",
            "Read its Abstract/Introduction/Related Work/References for outgoing references.",
            "Scan every other provided paper for exact or short-title mentions of the target paper.",
            "Only keep citation edges among the provided paper titles.",
        ],
        "target_terms": terms[:30],
        "ranked_docs": sorted(ranked_docs, key=lambda item: item["score"], reverse=True),
    }


def financial_reading_plan(record: Dict[str, Any], docs: Sequence[StructuredDoc]) -> Dict[str, Any]:
    question = normalize_ws(record.get("question", ""))
    question_terms = infer_financial_terms(question)
    company_terms = [name for name in record.get("doc", []) if name and lower_count(question, name)]
    terms = unique_terms(question_terms)

    ranked_docs = []
    focus_doc_ids = []
    for doc in docs:
        score = 0.0
        reasons = []
        title_blob = f"{doc.source_name} {doc.bundle_title}"
        if any(lower_count(title_blob, company) for company in company_terms):
            score += 20
            reasons.append("question-company-matches-doc")
            focus_doc_ids.append(doc.doc_id)
        if any(lower_count(doc.content[:2500], company) for company in company_terms):
            score += 8
            reasons.append("company-in-filing-front-matter")
            if doc.doc_id not in focus_doc_ids:
                focus_doc_ids.append(doc.doc_id)
        metric_hits = sum(literal_count(doc.content, term) for term in question_terms)
        if metric_hits:
            score += min(20, metric_hits * 2)
            reasons.append(f"metric-term-hits:{metric_hits}")
        if "10-q" in doc.content[:2000].casefold() or "10-k" in doc.content[:2000].casefold():
            score += 2
            reasons.append("sec-filing")
        ranked_docs.append({"doc_id": doc.doc_id, "title": doc.display_title, "score": round(score, 2), "reasons": reasons})

    return {
        "strategy": [
            "Route to the company named in the question.",
            "Within that filing, jump to financial statements and notes.",
            "Search for the accounting phrase and close variants.",
            "Read the nearest table row/column labels before extracting a value.",
        ],
        "target_terms": terms[:30],
        "routing_terms": company_terms,
        "focus_doc_ids": focus_doc_ids,
        "ranked_docs": sorted(ranked_docs, key=lambda item: item["score"], reverse=True),
    }


def legal_reading_plan(record: Dict[str, Any], docs: Sequence[StructuredDoc]) -> Dict[str, Any]:
    question = normalize_ws(record.get("question", ""))
    labels = infer_legal_labels(question)
    terms = unique_terms(labels + [term for label in labels for term in LEGAL_LABEL_TERMS.get(label, ())] + list(LEGAL_MARKERS))

    ranked_docs = []
    for doc in docs:
        guess = guess_legal_label(doc)
        label_scores = guess.get("scores", {})
        score = sum(label_scores.values())
        reasons = []
        if guess.get("label"):
            reasons.append(f"heuristic-label:{guess['label']}")
        marker_hits = sum(doc.content.count(marker) for marker in LEGAL_MARKERS)
        if marker_hits:
            reasons.append(f"legal-marker-hits:{marker_hits}")
            score += marker_hits
        ranked_docs.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.display_title,
                "score": round(float(score), 2),
                "reasons": reasons,
                "label_scores": label_scores,
            }
        )

    return {
        "strategy": [
            "Treat every judgment as a candidate because the task asks to classify all provided documents.",
            "Use title/case-cause/front matter as the first routing signal.",
            "Confirm with body markers such as applicant/executor/compensation/civil retrial terms.",
            "Return the provided judgment titles, not inferred party names.",
        ],
        "target_terms": terms[:40],
        "ranked_docs": sorted(ranked_docs, key=lambda item: item["doc_id"]),
    }


def build_reading_plan(record: Dict[str, Any], docs: Sequence[StructuredDoc]) -> Dict[str, Any]:
    doc_type = str(record.get("type", "")).lower()
    if doc_type == "paper":
        return paper_reading_plan(record, docs)
    if doc_type == "financial":
        return financial_reading_plan(record, docs)
    if doc_type == "legal":
        return legal_reading_plan(record, docs)
    return {"strategy": ["Use question terms to rank documents and sections."], "target_terms": english_tokens(record.get("question", "")), "ranked_docs": []}


def targeted_sections(doc: StructuredDoc, doc_type: str, target_terms: Sequence[str], limit: int = 4) -> List[Dict[str, Any]]:
    scored = []
    fallback_terms = target_terms
    for section in doc.sections:
        score, reasons = score_section(section, fallback_terms, doc_type)
        if score > 0:
            scored.append((score, reasons, section))
    if not scored:
        for section in doc.sections[:2]:
            scored.append((0.1, ["front-fallback"], section))
    scored.sort(key=lambda item: item[0], reverse=True)

    output = []
    for score, reasons, section in scored[:limit]:
        snippets = extract_snippets(section.text, fallback_terms, limit=3)
        if not snippets:
            chunks = chunk_text(section.text, target_chars=700)
            snippets = [{"term": "section-start", "snippet": compact(chunks[0] if chunks else section.text, 620)}]
        output.append(
            {
                "section_id": section.section_id,
                "heading": compact(section.heading, 100),
                "score": round(score, 2),
                "reasons": reasons,
                "snippets": snippets,
            }
        )
    return output


def structure_record(record: Dict[str, Any], selected_index: int, max_targeted_docs: int = 4) -> Dict[str, Any]:
    docs = parse_docs_bundle(record.get("docs", ""), record.get("doc", []) or [])
    doc_type = str(record.get("type", "")).lower()
    for doc in docs:
        doc.sections = split_sections(doc, doc_type)

    plan = build_reading_plan(record, docs)
    ranked_doc_ids = [item["doc_id"] for item in plan.get("ranked_docs", [])]
    if doc_type == "legal":
        selected_docs = docs
    elif plan.get("focus_doc_ids"):
        focus_set = set(plan.get("focus_doc_ids") or [])
        selected_docs = [doc for doc in docs if doc.doc_id in focus_set]
    else:
        selected_set = set(ranked_doc_ids[:max_targeted_docs])
        selected_docs = [doc for doc in docs if doc.doc_id in selected_set] or docs[:max_targeted_docs]

    target_terms = plan.get("target_terms", [])
    doc_summaries = []
    targeted = []
    total_sections = 0
    total_chunks = 0
    for doc in docs:
        chunks = sum(max(1, math.ceil(section.char_count / 1400)) for section in doc.sections)
        total_sections += len(doc.sections)
        total_chunks += chunks
        summary = {
            "doc_id": doc.doc_id,
            "title": doc.display_title,
            "bundle_title": doc.bundle_title,
            "source_name": doc.source_name,
            "chars": doc.char_count,
            "sections": len(doc.sections),
            "estimated_chunks": chunks,
            "section_inventory": section_inventory(doc),
        }
        if doc_type == "legal":
            summary["legal_label_guess"] = guess_legal_label(doc)
        doc_summaries.append(summary)

    for doc in selected_docs:
        targeted.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.display_title,
                "target_sections": targeted_sections(doc, doc_type, target_terms),
            }
        )

    return {
        "selected_index": selected_index,
        "record_id": record.get("id"),
        "set": record.get("set"),
        "type": record.get("type"),
        "level": record.get("level"),
        "language": record.get("language"),
        "question": record.get("question"),
        "instruction_preview": compact(record.get("instruction"), 360),
        "answer_preview": record.get("answer"),
        "doc_count": len(docs),
        "total_sections": total_sections,
        "estimated_chunks": total_chunks,
        "docs": doc_summaries,
        "reading_plan": plan,
        "targeted_reading": targeted,
    }


def summarize_structured(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_type = defaultdict(list)
    for item in records:
        by_type[item["type"]].append(item)

    summary = {"total_records": len(records), "by_type": {}}
    for doc_type, items in sorted(by_type.items()):
        summary["by_type"][doc_type] = {
            "records": len(items),
            "avg_docs": round(sum(item["doc_count"] for item in items) / len(items), 2),
            "avg_sections": round(sum(item["total_sections"] for item in items) / len(items), 2),
            "avg_estimated_chunks": round(sum(item["estimated_chunks"] for item in items) / len(items), 2),
            "levels": dict(sorted(Counter(item["level"] for item in items).items())),
            "languages": dict(sorted(Counter(item["language"] for item in items).items())),
        }
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_markdown(summary: Dict[str, Any], samples: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Loong Set1 Structured Reading Demo",
        "",
        "## Dataset summary",
        "",
        f"- total structured records: {summary['total_records']}",
    ]
    for doc_type, info in summary["by_type"].items():
        lines.append(
            f"- {doc_type}: records={info['records']}, avg_docs={info['avg_docs']}, "
            f"avg_sections={info['avg_sections']}, avg_chunks={info['avg_estimated_chunks']}, "
            f"levels={info['levels']}, languages={info['languages']}"
        )

    for sample in samples:
        lines.extend(
            [
                "",
                f"## Sample: {sample['type']} idx={sample['selected_index']} level={sample['level']}",
                "",
                f"- record_id: `{sample['record_id']}`",
                f"- question: {compact(sample.get('question'), 420)}",
                f"- docs/sections/chunks: {sample['doc_count']} / {sample['total_sections']} / {sample['estimated_chunks']}",
                "- strategy:",
            ]
        )
        for step in sample["reading_plan"].get("strategy", []):
            lines.append(f"  - {step}")
        ranked = sample["reading_plan"].get("ranked_docs", [])[:5]
        lines.append("- top document routing:")
        for doc in ranked:
            reason = ", ".join(doc.get("reasons", [])[:4]) or "no strong signal"
            lines.append(f"  - {doc['doc_id']} score={doc['score']}: {compact(doc['title'], 100)} ({reason})")
        lines.append("- targeted snippets:")
        for targeted_doc in sample.get("targeted_reading", [])[:3]:
            lines.append(f"  - {targeted_doc['doc_id']}: {compact(targeted_doc['title'], 110)}")
            for section in targeted_doc.get("target_sections", [])[:2]:
                reasons = ", ".join(section.get("reasons", [])[:4])
                lines.append(f"    - {section['section_id']} `{section['heading']}` score={section['score']} reasons={reasons}")
                for snippet in section.get("snippets", [])[:1]:
                    lines.append(f"      - {compact(snippet['snippet'], 320)}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--set-id", type=int, default=1)
    parser.add_argument("--types", nargs="+", default=["paper", "financial", "legal"])
    parser.add_argument("--sample-per-type", type=int, default=1)
    parser.add_argument("--max-targeted-docs", type=int, default=4)
    parser.add_argument("--write-full-jsonl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted_types = {item.lower() for item in args.types}
    raw_records = load_jsonl(args.input_path)

    structured: List[Dict[str, Any]] = []
    for idx, record in enumerate(raw_records):
        if int(record.get("set", 0) or 0) != args.set_id:
            continue
        if str(record.get("type", "")).lower() not in wanted_types:
            continue
        structured.append(structure_record(record, idx, max_targeted_docs=args.max_targeted_docs))

    summary = summarize_structured(structured)
    samples: List[Dict[str, Any]] = []
    seen_counts = Counter()
    for item in structured:
        doc_type = item["type"]
        if seen_counts[doc_type] < args.sample_per_type:
            samples.append(item)
            seen_counts[doc_type] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "samples.json", samples)
    write_json(args.output_dir / "preview.json", {"summary": summary, "samples": samples})
    (args.output_dir / "preview.md").write_text(render_markdown(summary, samples), encoding="utf-8")
    if args.write_full_jsonl:
        write_jsonl(args.output_dir / "structured_records.jsonl", structured)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote: {args.output_dir / 'preview.md'}")
    print(f"Wrote: {args.output_dir / 'preview.json'}")
    if args.write_full_jsonl:
        print(f"Wrote: {args.output_dir / 'structured_records.jsonl'}")


if __name__ == "__main__":
    main()
