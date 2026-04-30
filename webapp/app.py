"""LAMBO Pipeline Visualizer — Streamlit web UI.

Reads `logs/lambo_v2_toc_99/` outputs and renders, for any chosen sample:
  • the question / instruction / gold answer / model answer
  • each document organised under its hierarchical TOC, with section text
    placed under the anchor's section heading (MODORA-style document view)
  • DocRefineAgent's <think>/<search>/<info> trace, with opened sections
    flagged on the left panel
  • the composer/generator outputs and per-sample LLM-judge verdict
  • evidence spans highlighted across the matching section text in HTML
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "lambo_v2_toc_99"


# ---------------------------------------------------------------------------
# Streamlit page config + CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LAMBO Pipeline Visualizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --bg:        #0f1115;
  --surface:   #161922;
  --surface-2: #1d2230;
  --border:    #262b3a;
  --muted:     #8a93a6;
  --text:      #e6e9f2;
  --accent:    #8b5cf6;
  --accent-2:  #06b6d4;
  --good:      #34d399;
  --warn:      #fbbf24;
  --bad:       #f87171;
  --highlight-bg: rgba(251, 191, 36, 0.28);
  --highlight-fg: #fde68a;
}

.stApp {
  background:
    radial-gradient(1100px 600px at 80% -10%, rgba(139,92,246,0.18), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(6,182,212,0.16), transparent 60%),
    var(--bg);
  color: var(--text);
}

section[data-testid="stSidebar"] > div { background: var(--surface); border-right: 1px solid var(--border); }

h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.01em; }

.lambo-header {
  display:flex; align-items:center; gap:12px;
  padding:14px 18px; border-radius:14px;
  background:linear-gradient(135deg, rgba(139,92,246,0.18), rgba(6,182,212,0.10));
  border:1px solid var(--border); margin-bottom:18px;
}
.lambo-header .logo {
  width:38px; height:38px; border-radius:10px;
  background:linear-gradient(135deg,#8b5cf6,#06b6d4);
  display:flex; align-items:center; justify-content:center;
  font-weight:800; color:white; font-size:18px;
}
.lambo-header .title { font-size:20px; font-weight:700; }
.lambo-header .subtitle { color:var(--muted); font-size:13px; }

.card { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:16px 18px; margin-bottom:14px; }
.card h4 { margin:0 0 10px 0; font-size:14px; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; }

.kvgrid { display:grid; grid-template-columns: 130px 1fr; row-gap:6px; column-gap:14px; }
.kvgrid .k { color:var(--muted); font-size:12px; padding-top:3px; }
.kvgrid .v { color:var(--text); font-size:14px; }

.qa-block { padding:12px 14px; background:var(--surface-2); border-radius:10px; border:1px solid var(--border); font-size:14px; line-height:1.55; white-space:pre-wrap; word-break:break-word; }
.qa-block.gold { border-left: 3px solid var(--good); }
.qa-block.pred { border-left: 3px solid var(--accent); }

.badge { display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px; font-weight:600; line-height:1.6; border:1px solid var(--border); background:var(--surface-2); color:var(--muted); }
.badge.good { background:rgba(52,211,153,0.12); border-color:rgba(52,211,153,0.4); color:var(--good); }
.badge.warn { background:rgba(251,191,36,0.12); border-color:rgba(251,191,36,0.4); color:var(--warn); }
.badge.bad  { background:rgba(248,113,113,0.12); border-color:rgba(248,113,113,0.4); color:var(--bad); }
.badge.acc  { background:rgba(139,92,246,0.14); border-color:rgba(139,92,246,0.4); color:#c4b5fd; }

.doc-header { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; background:var(--surface-2); border:1px solid var(--border); border-radius:10px; margin: 10px 0 6px; }
.doc-header .doc-title { font-size:15px; font-weight:600; }
.doc-header .doc-meta { color:var(--muted); font-size:12px; }

.toc-section { border-left:2px solid var(--border); margin-left:2px; padding:6px 0 6px 14px; }
.toc-section.opened { border-left-color: var(--accent); }
.toc-heading { display:flex; align-items:center; gap:8px; font-weight:600; color:var(--text); }
.toc-heading .num { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color:var(--muted); font-size:12px; }
.toc-heading .title { font-size:14px; }
.toc-heading .lv { color:var(--muted); font-size:11px; }
.toc-text { margin:6px 0 4px 0; padding:10px 12px; background:rgba(255,255,255,0.02); border:1px solid var(--border); border-radius:8px; font-size:13px; line-height:1.65; white-space:pre-wrap; color:#cdd3e3; max-height: 360px; overflow:auto; }
.toc-text mark,
.trace-block mark { background:var(--highlight-bg); color:var(--highlight-fg); padding:1px 3px; border-radius:3px; box-shadow:0 0 0 1px rgba(251,191,36,0.4) inset; }

.trace-block { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12.5px; line-height:1.55; background:#0b0d12; border:1px solid var(--border); border-radius:10px; padding:14px 16px; white-space:pre-wrap; word-break:break-word; max-height: 520px; overflow:auto; color:#cbd5e1; }
.trace-block .think  { color:#7dd3fc; }
.trace-block .search { color:#fcd34d; }
.trace-block .info   { color:#a5f3fc; }
.trace-block .answer { color:#86efac; }

.pill-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.pill { padding:3px 9px; border-radius:999px; font-size:12px; background:var(--surface-2); color:var(--muted); border:1px solid var(--border); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.pill.opened { background:rgba(139,92,246,0.15); color:#c4b5fd; border-color:rgba(139,92,246,0.4); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_samples(log_dir: Path) -> List[str]:
    samples_dir = log_dir / "samples"
    if not samples_dir.exists():
        return []
    return sorted(p.name for p in samples_dir.iterdir() if p.is_dir())


def load_manifest_index(log_dir: Path) -> Dict[str, Dict[str, Any]]:
    raw = load_json(log_dir / "manifest.json") or []
    return {str(item.get("sample_id")): item for item in raw if isinstance(item, dict)}


def load_predictions_index(log_dir: Path) -> Dict[str, Dict[str, Any]]:
    pred_path = log_dir / "lambo_predictions.jsonl"
    out: Dict[str, Dict[str, Any]] = {}
    if not pred_path.exists():
        return out
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                out[str(row.get("sample_id"))] = row
            except Exception:
                continue
    return out


def load_judge_index(log_dir: Path) -> Dict[str, Dict[str, Any]]:
    judge = load_json(log_dir / "reports" / "llm_judge.json") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for v in judge.get("verdicts", []):
        if isinstance(v, dict):
            out[str(v.get("sample_id"))] = v
    return out


def load_sample_artifacts(log_dir: Path, sample_id: str) -> Dict[str, Any]:
    sample_dir = log_dir / "samples" / sample_id
    anchors = load_json(sample_dir / "anchors_v2.json")
    if anchors is None:
        anchors = load_json(sample_dir / "anchors_v2.partial.json")
    composed = load_json(sample_dir / "composed_v2.json") or load_json(sample_dir / "composed_v3.json")
    generator = load_json(sample_dir / "generator.json")
    error = load_json(sample_dir / "error.json")

    refines: Dict[str, Dict[str, Any]] = {}
    if sample_dir.exists():
        for p in sample_dir.glob("DOC*_refine.json"):
            data = load_json(p)
            if data:
                doc_id = data.get("doc_id") or p.stem.replace("_refine", "")
                refines[doc_id] = data
    return {
        "sample_dir": sample_dir,
        "anchors": anchors,
        "composed": composed,
        "generator": generator,
        "error": error,
        "refines": refines,
    }


# ---------------------------------------------------------------------------
# Evidence parsing & highlighting
# ---------------------------------------------------------------------------
def _flatten_strings(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        if obj:
            out.append(obj)
    elif isinstance(obj, list):
        for x in obj:
            out.extend(_flatten_strings(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_strings(v))
    return out


def parse_evidence_spans(evidence: Any) -> List[str]:
    """Normalise the doc_refine `evidence` field. It can be a JSON list,
    a JSON dict (whose string values are the spans), or a raw string that
    *looks* like JSON but is broken because the LLM left unescaped quotes
    inside table-cell evidence. Try every reasonable fallback."""
    if not evidence:
        return []
    if isinstance(evidence, (list, dict)):
        return _flatten_strings(evidence)
    if not isinstance(evidence, str):
        return [str(evidence)]

    s = evidence.strip()
    # 1) strict JSON
    try:
        return _flatten_strings(json.loads(s))
    except Exception:
        pass
    # 2) Python literal — handles single-quoted, etc.
    try:
        import ast
        return _flatten_strings(ast.literal_eval(s))
    except Exception:
        pass
    # 3) broken JSON list — split on "," delimiters between quoted strings
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        chunks = re.split(r'"\s*,\s*"', inner)
        cleaned = [c.strip().strip('"').strip("'") for c in chunks]
        cleaned = [c for c in cleaned if c]
        if cleaned:
            return cleaned
        return [inner]
    # 4) plain string
    return [s]


def dedupe_spans(spans: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for span in spans:
        if not isinstance(span, str):
            span = str(span)
        span = span.strip()
        if not span:
            continue
        key = re.sub(r"\s+", " ", span)
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return out


def parse_answer_spans_from_trace(trace_text: str) -> List[str]:
    """Extract evidence-like spans emitted inside DocRefine <answer> tags.

    Some runs store clean evidence in `refine.evidence`; others only preserve
    the final evidence payload in the raw trace. Treat <answer> as another
    structured-evidence source so the UI can point back to the original <info>
    text that produced it.
    """
    if not trace_text:
        return []
    spans: List[str] = []
    for match in re.finditer(r"<answer>([\s\S]*?)</answer>", trace_text, flags=re.IGNORECASE):
        payload = match.group(1).strip()
        spans.extend(parse_evidence_spans(payload))
    return dedupe_spans(spans)


def get_refine_evidence_spans(refine: Dict[str, Any]) -> List[str]:
    spans = parse_evidence_spans(refine.get("evidence", ""))
    spans.extend(parse_answer_spans_from_trace(refine.get("trace", "")))
    return dedupe_spans(spans)


# Canonical character classes — LLM evidence spans often replace
# typographic quotes/dashes with ASCII equivalents (or vice versa), so we
# match in a normalised space and translate matches back to original offsets.
_QUOTE_CHARS = [
    chr(0x22), chr(0x27), chr(0x60), chr(0xB4),
    chr(0x2018), chr(0x2019), chr(0x201C), chr(0x201D),
    chr(0x00AB), chr(0x00BB),
]
_DASH_CHARS = [chr(0x2D), chr(0x2013), chr(0x2014), chr(0x2212)]
_CHAR_CANON: Dict[str, str] = {}
for _ch in _QUOTE_CHARS:
    _CHAR_CANON[_ch] = '"'
for _ch in _DASH_CHARS:
    _CHAR_CANON[_ch] = '-'
_CHAR_CANON[chr(0xA0)] = ' '   # non-breaking space


def _canon_char(c: str) -> str:
    if c in _CHAR_CANON:
        return _CHAR_CANON[c]
    if c.isspace():
        return " "
    return c.lower()


def _build_canon(text: str) -> Tuple[str, List[int]]:
    """Return (canonical_text, mapping). mapping[i] = original_index of
    canonical_text[i]. Whitespace runs collapse to a single space; the map
    points to the FIRST character of the original whitespace run."""
    canon: List[str] = []
    mapping: List[int] = []
    prev_space = False
    for i, c in enumerate(text):
        cc = _canon_char(c)
        if cc == " ":
            if prev_space:
                continue
            canon.append(" ")
            mapping.append(i)
            prev_space = True
        else:
            canon.append(cc)
            mapping.append(i)
            prev_space = False
    return "".join(canon), mapping


def _canon_span(span: str) -> str:
    canon: List[str] = []
    prev_space = False
    for c in span:
        cc = _canon_char(c)
        if cc == " ":
            if prev_space:
                continue
            canon.append(" ")
            prev_space = True
        else:
            canon.append(cc)
            prev_space = False
    return "".join(canon).strip()


def highlight_spans(section_text: str, spans: List[str]) -> str:
    if not section_text:
        return ""
    canon_text, mapping = _build_canon(section_text)
    ranges: List[Tuple[int, int]] = []
    for span in spans:
        if not span or len(span) < 6:
            continue
        cspan = _canon_span(span)
        if not cspan:
            continue
        # Try full canonical substring match first.
        start = 0
        any_hit = False
        while True:
            idx = canon_text.find(cspan, start)
            if idx < 0:
                break
            any_hit = True
            end_canon = idx + len(cspan)
            orig_start = mapping[idx]
            orig_end = (
                mapping[end_canon] if end_canon < len(mapping) else len(section_text)
            )
            ranges.append((orig_start, orig_end))
            start = end_canon
        if any_hit:
            continue
        # Fallback: locate the longest contiguous prefix that matches.
        # Useful when the evidence span is slightly truncated/extended.
        for prefix_len in (160, 120, 80, 60, 40):
            if len(cspan) < prefix_len:
                continue
            prefix = cspan[:prefix_len]
            idx = canon_text.find(prefix)
            if idx >= 0:
                end_canon = idx + len(cspan) if idx + len(cspan) <= len(canon_text) else idx + prefix_len
                orig_start = mapping[idx]
                orig_end = (
                    mapping[end_canon] if end_canon < len(mapping) else len(section_text)
                )
                ranges.append((orig_start, orig_end))
                break
    if not ranges:
        return html.escape(section_text)

    ranges.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in ranges:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    out: List[str] = []
    cursor = 0
    for s, e in merged:
        out.append(html.escape(section_text[cursor:s]))
        out.append(f"<mark>{html.escape(section_text[s:e])}</mark>")
        cursor = e
    out.append(html.escape(section_text[cursor:]))
    return "".join(out)


def render_trace_segment(trace_text: str) -> str:
    safe = html.escape(trace_text)
    safe = re.sub(r"&lt;think&gt;([\s\S]*?)&lt;/think&gt;",
                  r'<span class="think">&lt;think&gt;\1&lt;/think&gt;</span>', safe)
    safe = re.sub(r"&lt;search&gt;([\s\S]*?)&lt;/search&gt;",
                  r'<span class="search">&lt;search&gt;\1&lt;/search&gt;</span>', safe)
    safe = re.sub(r"&lt;answer&gt;([\s\S]*?)&lt;/answer&gt;",
                  r'<span class="answer">&lt;answer&gt;\1&lt;/answer&gt;</span>', safe)
    return safe


def render_trace(trace_text: str, evidence_spans: Optional[List[str]] = None) -> str:
    if not trace_text:
        return "(no trace)"

    evidence_spans = evidence_spans or []
    info_re = re.compile(
        r"<info\s+anchor_id=(['\"])(.*?)\1>([\s\S]*?)</info>",
        flags=re.IGNORECASE,
    )
    out: List[str] = []
    cursor = 0
    for match in info_re.finditer(trace_text):
        out.append(render_trace_segment(trace_text[cursor:match.start()]))
        anchor_id = match.group(2)
        info_body = match.group(3)
        info_html = highlight_spans(info_body, evidence_spans)
        out.append(
            '<span class="info">'
            f'&lt;info anchor_id="{html.escape(anchor_id)}"&gt;'
            f"{info_html}"
            "&lt;/info&gt;"
            "</span>"
        )
        cursor = match.end()
    out.append(render_trace_segment(trace_text[cursor:]))
    return "".join(out)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="lambo-header" style="margin:6px 0 14px 0;">
      <div class="logo">L</div>
      <div>
        <div class="title">LAMBO</div>
        <div class="subtitle">Pipeline Visualizer</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

log_dir_str = st.sidebar.text_input(
    "Experiment log directory",
    value=str(DEFAULT_LOG_DIR),
    help="Path to a logs/<experiment_name>/ dir containing samples/ and reports/.",
)
log_dir = Path(log_dir_str)

samples = list_samples(log_dir)
if not samples:
    st.error(f"No samples found under: {log_dir / 'samples'}")
    st.stop()

manifest_idx = load_manifest_index(log_dir)
predictions_idx = load_predictions_index(log_dir)
judge_idx = load_judge_index(log_dir)


def sample_label(sid: str) -> str:
    j = judge_idx.get(sid, {})
    score = j.get("score")
    score_part = f"  ·  {score:g}" if isinstance(score, (int, float)) else ""
    return f"{sid}{score_part}"


sample_id = st.sidebar.selectbox("Sample", samples, format_func=sample_label, index=0)

show_distractors = st.sidebar.checkbox("Show all sections (not just opened)", value=True)
expand_text = st.sidebar.checkbox("Expand section text", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='color:var(--muted); font-size:12px;'>"
    f"{len(samples)} samples · {sum(1 for s in samples if s in judge_idx)} judged"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
artifacts = load_sample_artifacts(log_dir, sample_id)
manifest_item = manifest_idx.get(sample_id, {})
prediction = predictions_idx.get(sample_id, {})
judge_verdict = judge_idx.get(sample_id, {})

st.markdown(
    f"""
    <div class="lambo-header">
      <div class="logo">L</div>
      <div>
        <div class="title">{html.escape(sample_id)}</div>
        <div class="subtitle">
          {html.escape(manifest_item.get("record_type", prediction.get("type", "—")))}
          · level {manifest_item.get("level", prediction.get("level", "—"))}
          · idx {manifest_item.get("selected_index", prediction.get("selected_index", "—"))}
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Top: Question, Instruction, Gold, Prediction, Verdict ----
score = judge_verdict.get("score")
score_badge_cls = "bad"
if isinstance(score, (int, float)):
    score_badge_cls = "good" if score >= 80 else ("warn" if score >= 50 else "bad")

q_col, a_col = st.columns([1.2, 1])

with q_col:
    st.markdown('<div class="card"><h4>Question</h4>', unsafe_allow_html=True)
    question = manifest_item.get("question") or prediction.get("question") or "—"
    st.markdown(f'<div class="qa-block">{html.escape(str(question))}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><h4>Instruction</h4>', unsafe_allow_html=True)
    instruction = prediction.get("instruction") or "—"
    st.markdown(f'<div class="qa-block">{html.escape(str(instruction))}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with a_col:
    st.markdown(
        f'<div class="card"><h4>LLM Judge</h4>'
        f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">'
        f'<span class="badge {score_badge_cls}">'
        f'{("score " + str(score)) if score is not None else "no verdict"}'
        f'</span>'
        f'<span class="badge acc">type {html.escape(str(judge_verdict.get("type","—")))}</span>'
        f'<span class="badge acc">level {html.escape(str(judge_verdict.get("level","—")))}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card"><h4>Gold Answer</h4>', unsafe_allow_html=True)
    gold = judge_verdict.get("gold") or prediction.get("answer") or "—"
    st.markdown(f'<div class="qa-block gold">{html.escape(str(gold))}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><h4>Model Answer</h4>', unsafe_allow_html=True)
    pred = (
        (artifacts.get("generator") or {}).get("final_answer")
        or judge_verdict.get("prediction")
        or prediction.get("generate_response")
        or "—"
    )
    if not isinstance(pred, str):
        pred = json.dumps(pred, ensure_ascii=False, indent=2)
    st.markdown(f'<div class="qa-block pred">{html.escape(str(pred))}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---- Tabs ----
tab_docs, tab_trace, tab_pipeline = st.tabs(
    ["📄 Document Browser", "🔍 Search Trace", "🧩 Composer / Generator"]
)


# ===== TAB 1: Document Browser ===============================================
with tab_docs:
    if artifacts["error"] is not None:
        st.error(f"Sample errored: {artifacts['error'].get('error', '?')}")
    if artifacts["anchors"] is None:
        st.warning("No anchors_v2.json (or partial) found for this sample.")
    else:
        for doc_payload in artifacts["anchors"].get("docs", []):
            doc_id = doc_payload.get("doc_id", "DOC?")
            doc_title = doc_payload.get("doc_title", "(untitled)")
            toc = doc_payload.get("toc", []) or []

            refine = artifacts["refines"].get(doc_id, {})
            opened = set(str(a) for a in refine.get("opened_anchors", []))
            scan = refine.get("scan_result", "—")
            evidence_spans = get_refine_evidence_spans(refine)

            scan_cls = (
                "good" if scan == "evidence_found" else ("warn" if scan == "no_evidence" else "bad")
            )

            st.markdown(
                f"""
                <div class="doc-header">
                  <div>
                    <div class="doc-title">{html.escape(doc_id)} · {html.escape(doc_title)}</div>
                    <div class="doc-meta">
                      {len(toc)} sections · opened {len(opened)} ·
                      rounds {refine.get("rounds_used", 0)} ·
                      evidence spans {len(evidence_spans)}
                    </div>
                  </div>
                  <span class="badge {scan_cls}">{html.escape(str(scan))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if opened:
                pills = "".join(
                    f'<span class="pill opened">{html.escape(str(a))}</span>'
                    for a in sorted(opened)
                )
                st.markdown(f'<div class="pill-row">{pills}</div>', unsafe_allow_html=True)

            for entry in toc:
                num = str(entry.get("number", ""))
                title = str(entry.get("title", ""))
                level = int(entry.get("level", 1))
                section_text = entry.get("text", "") or ""
                is_opened = num in opened

                if not show_distractors and not is_opened:
                    continue

                indent_px = (level - 1) * 18
                opened_cls = "opened" if is_opened else ""

                heading_html = f"""
                <div class="toc-section {opened_cls}" style="margin-left:{indent_px}px;">
                  <div class="toc-heading">
                    <span class="num">[{html.escape(num)}]</span>
                    <span class="title">{html.escape(title)}</span>
                    <span class="lv">L{level}</span>
                """
                if is_opened:
                    heading_html += '<span class="badge acc">opened</span>'
                heading_html += "</div>"
                st.markdown(heading_html, unsafe_allow_html=True)

                if expand_text and section_text:
                    text_html = highlight_spans(
                        section_text, evidence_spans if is_opened else []
                    )
                    st.markdown(f'<div class="toc-text">{text_html}</div>', unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)


# ===== TAB 2: Search Trace ===================================================
with tab_trace:
    if not artifacts["refines"]:
        st.warning("No DocRefine traces found for this sample.")
    else:
        cols = st.columns(len(artifacts["refines"]))
        for col, (doc_id, refine) in zip(cols, sorted(artifacts["refines"].items())):
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                      <h4>{html.escape(doc_id)} · {html.escape(refine.get("doc_title",""))[:40]}</h4>
                      <div class="kvgrid">
                        <div class="k">scan</div>
                        <div class="v">{html.escape(str(refine.get("scan_result","")))}</div>
                        <div class="k">rounds</div>
                        <div class="v">{refine.get("rounds_used", 0)}</div>
                        <div class="k">opened</div>
                        <div class="v" style="font-family:ui-monospace,monospace;font-size:12px;">
                          {", ".join(html.escape(str(a)) for a in refine.get("opened_anchors", []))}
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="trace-block">{render_trace(refine.get("trace",""), get_refine_evidence_spans(refine))}</div>',
                    unsafe_allow_html=True,
                )


# ===== TAB 3: Composer / Generator ===========================================
with tab_pipeline:
    composed = artifacts["composed"] or {}
    generator = artifacts["generator"] or {}

    cl, cr = st.columns(2)

    with cl:
        st.markdown('<div class="card"><h4>Composer</h4>', unsafe_allow_html=True)
        if not composed:
            st.markdown("(no composed_v*.json)", unsafe_allow_html=True)
        else:
            qspec = composed.get("query_spec")
            if isinstance(qspec, dict):
                proj = (qspec.get("projector") or {})
                st.markdown(
                    f"""
                    <div class="kvgrid">
                      <div class="k">intent</div>     <div class="v">{html.escape(str(qspec.get("intent","—")))}</div>
                      <div class="k">selector</div>   <div class="v">{html.escape(str(qspec.get("selector","—")))}</div>
                      <div class="k">relator_kind</div><div class="v">{html.escape(str(qspec.get("relator_kind","—")))}</div>
                      <div class="k">ref_unit</div>   <div class="v">{html.escape(str(proj.get("ref_unit","—")))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            structure = composed.get("structure") or {}
            if structure:
                st.markdown(
                    f'<div class="kvgrid"><div class="k">structure</div>'
                    f'<div class="v">{html.escape(str(structure.get("form","—")))} — '
                    f'{html.escape(str(structure.get("summary","")))}</div></div>',
                    unsafe_allow_html=True,
                )

            proj_map = composed.get("projection_map") or {}
            if proj_map:
                st.markdown("<h4 style='margin-top:14px;'>Projection map</h4>", unsafe_allow_html=True)
                rows = "".join(
                    f'<div class="k">{html.escape(str(k))}</div><div class="v">{html.escape(str(v))}</div>'
                    for k, v in proj_map.items()
                )
                st.markdown(f'<div class="kvgrid">{rows}</div>', unsafe_allow_html=True)

            records = composed.get("records") or composed.get("doc_records") or []
            if records:
                st.markdown("<h4 style='margin-top:14px;'>Records</h4>", unsafe_allow_html=True)
                st.code(json.dumps(records, ensure_ascii=False, indent=2), language="json")

            filled = composed.get("filled_skeleton")
            if filled is not None:
                st.markdown("<h4 style='margin-top:14px;'>Filled Skeleton (v3)</h4>", unsafe_allow_html=True)
                st.code(
                    json.dumps(filled, ensure_ascii=False, indent=2)
                    if not isinstance(filled, str) else filled,
                    language="json",
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with cr:
        st.markdown('<div class="card"><h4>Generator</h4>', unsafe_allow_html=True)
        if not generator:
            st.markdown("(no generator.json)", unsafe_allow_html=True)
        else:
            final = generator.get("final_answer")
            if not isinstance(final, str):
                final = json.dumps(final, ensure_ascii=False, indent=2)
            st.markdown(f'<div class="qa-block pred">{html.escape(final)}</div>', unsafe_allow_html=True)

            raw = generator.get("raw_text") or ""
            if raw and raw != final:
                with st.expander("raw LLM text"):
                    st.code(raw)

            ru = generator.get("ref_unit")
            if ru:
                st.markdown(
                    f'<div class="kvgrid" style="margin-top:10px;"><div class="k">ref_unit</div><div class="v">{html.escape(ru)}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    if judge_verdict.get("raw"):
        with st.expander("LLM judge rationale"):
            st.markdown(
                f'<div class="qa-block">{html.escape(str(judge_verdict.get("raw","")))}</div>',
                unsafe_allow_html=True,
            )
