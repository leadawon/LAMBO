# LAMBO — Long-document Agentic Multi-doc Benchmark Optimizer

Agentic RAG pipeline for the [Loong benchmark](https://github.com/Loong-Long-Document-QA), focused on long-document, multi-doc QA across **financial reports, legal cases, and academic papers**.

The repo carries multiple pipeline generations (`v1` → `v2` → `v2-toc` → `v3`) so each stage of the design is reproducible and ablatable.

---

## Table of Contents

- [Latest Pipeline (v3)](#latest-pipeline-v3)
- [Module-Level I/O](#module-level-io)
- [Pipeline Versions](#pipeline-versions)
- [Repository Layout](#repository-layout)
- [Experiment Logs Layout](#experiment-logs-layout)
- [Reproducing Experiments](#reproducing-experiments)
- [Environment](#environment)
- [Results Summary](#results-summary)

---

## Latest Pipeline (v3)

```
┌──────────────────────┐
│  AnchorAgentV2       │  Per-doc: line-numbered text → LLM emits hierarchical TOC
│                      │  (number, title, level, start_line, end_line)
│                      │  Code converts L-numbers → char_start/char_end deterministically
└──────────┬───────────┘
           │  doc_payload = {doc_id, doc_title, toc:[{number, title, level, text, ...}]}
           ▼
┌──────────────────────┐
│  DocRefineAgentV2    │  Per-doc iterative loop:
│                      │    <think> → <search action="open" anchor_id="..."> → <info>
│                      │  Output: evidence_sheet with verbatim spans grounded in TOC sections
└──────────┬───────────┘
           │  doc_sheet = {doc_id, doc_title, scan_result, evidence, opened_anchors, trace}
           ▼
┌──────────────────────┐
│  GlobalComposerV3    │  SINGLE LLM call → 5 sections in fixed order:
│                      │    1) query_spec       — intent / selector / relator_kind / projector
│                      │    2) doc_records      — verdict + ref_value + facts + outgoing_refs
│                      │    3) structure        — fan_in / chain / partition / total_order / ...
│                      │    4) completeness    — expected_hint vs selected_count
│                      │    5) filled_skeleton — candidate answer in target shape
│                      │  Domain-blind: no domain/language hints; only question + instruction
└──────────┬───────────┘
           │  composed = {query_spec, doc_records, structure, completeness, filled_skeleton}
           ▼
┌──────────────────────┐
│  GeneratorV2         │  Verifies filled_skeleton conforms to instruction format
│                      │  Reformats if needed; never edits ref_values or invents content
└──────────┬───────────┘
           │  final_answer
           ▼
┌──────────────────────┐
│  Evaluation          │  structured_eval (EM, pair-F1) + LLM judge (Qwen-as-judge, 1–100)
└──────────────────────┘
```

The v2 pipeline keeps the same first two stages but uses `GlobalComposerV2` + the original `Generator`. The v1 pipeline (`scripts/run_set1.py`) uses the older non-TOC `AnchorAgent` and `DocRefineAgent`.

---

## Module-Level I/O

### Stage 1 — `AnchorAgentV2` ([lambo_v2/agents/anchor_agent_v2.py](lambo_v2/agents/anchor_agent_v2.py))

| Field | Type | Notes |
|---|---|---|
| **Input** | | |
| `record` | dict | Loong record (`question`, `instruction`, `doc_text`, …) |
| `sample_dir` | Path | per-sample cache dir |
| **Output** (`anchors_v2.json`) | | |
| `docs[].doc_id` | str | `DOC1`, `DOC2`, … |
| `docs[].doc_title` | str | document title from header / heuristic |
| `docs[].toc[]` | list | hierarchical TOC entries |
| `docs[].toc[].number` | str | dotted section number (`"3.2.1"`) |
| `docs[].toc[].title` | str | section title |
| `docs[].toc[].level` | int | depth (1 = top-level) |
| `docs[].toc[].start_line` / `end_line` | str | LLM-emitted L-number anchors (`"L0042"`) |
| `docs[].toc[].char_start` / `char_end` | int | deterministic char offsets resolved by the agent |
| `docs[].toc[].text` | str | actual section body, sliced from char span |
| `docs[].toc_count` | int | number of sections |
| `segmentation_mode` | str | `"hierarchical_toc_line_anchored"` |

### Stage 2 — `DocRefineAgentV2` ([lambo_v2/agents/doc_refine_agent_v2.py](lambo_v2/agents/doc_refine_agent_v2.py))

| Field | Type | Notes |
|---|---|---|
| **Input** | | |
| `question`, `instruction` | str | raw user query and instruction |
| `doc_payload` | dict | one entry from AnchorAgentV2's `docs[]` |
| `other_docs` | list | `[{doc_id, doc_title}]` for sibling docs (cross-doc awareness) |
| **LLM loop tags** | | |
| `<think>` | LLM | reasoning over current TOC and previously opened sections |
| `<search>` | LLM | JSON `{"action": "open"|"stop", "anchor_id": "<dotted number>"}` |
| `<info>` | injected | section text retrieved by `anchor_id` |
| `<answer>` | LLM | verbatim evidence span(s) when LLM emits `action="stop"` |
| **Output** (`<doc_id>_refine.json`) | | |
| `doc_id`, `doc_title` | str | passthrough |
| `scan_result` | str | `evidence_found` / `no_evidence` |
| `evidence` | str | concatenated verbatim spans from `<answer>` |
| `opened_anchors` | list[str] | section numbers visited |
| `rounds_used` | int | LLM rounds consumed |
| `trace` | str | full `<think>/<search>/<info>` log |

### Stage 3 — `GlobalComposerV3` ([lambo_v2/agents/global_composer_v3.py](lambo_v2/agents/global_composer_v3.py))

| Field | Type | Notes |
|---|---|---|
| **Input** | | |
| `question`, `instruction` | str | passthrough |
| `doc_sheets` | list | DocRefineAgentV2 outputs for all docs |
| `doc_title_map` | dict | `{doc_id: doc_title}` |
| **Single-call LLM output** (`composed_v3.json`) | | |
| `query_spec.intent` | str | one-sentence task description |
| `query_spec.selector` | str | predicate that a matching doc must satisfy |
| `query_spec.relator_kind` | enum | `set / total_order / mapping / graph / sequence / none` |
| `query_spec.projector.ref_unit` | enum | `doc_id / entity_name / title / numeric / composite` |
| `query_spec.projector.ref_unit_justification` | str | trigger quote from instruction or question |
| `query_spec.projector.output_skeleton` | str/JSON | exact format the answer must conform to |
| `doc_records[].doc_id` | str | one entry per input doc, in original order |
| `doc_records[].verdict` | enum | `match / distractor / no_evidence` |
| `doc_records[].ref_value` | str | answer-surface string for this doc, formatted per `ref_unit` |
| `doc_records[].facts` | str | natural-language summary with verbatim quotes |
| `doc_records[].outgoing_refs[]` | list | `[{to_doc, kind, quote}]` cross-doc references |
| `structure.form` | enum | `fan_in / fan_out / linear_chain / tree / partition / total_order / sequence / flat_set / single` |
| `structure.summary` | str | one-sentence structural description |
| `completeness.expected_hint` | str | quote from instruction or `"open"` |
| `completeness.selected_count` | int | match-verdict count |
| `completeness.potentially_missing[]` | list | distractor doc ids that might still match |
| `filled_skeleton` | any | Composer's candidate answer in the instruction's exact shape |
| `warnings[]` | list | post-processing flags (e.g. `selected_count mismatch`) |
| _backward-compat fields_ | | `projection_map`, `records`, `structure_description`, `ref_unit` |

### Stage 4 — `GeneratorV2` ([lambo_v2/agents/generator_v2.py](lambo_v2/agents/generator_v2.py))

| Field | Type | Notes |
|---|---|---|
| **Input** | | composer's full handoff (query_spec, doc_records, structure, completeness, filled_skeleton) + question/instruction |
| **Output** (`generator.json`) | | |
| `final_answer` | str / JSON | the user-facing answer (parsed JSON if shape allows) |
| `raw_text` | str | LLM raw output |
| `filled_skeleton` | any | passthrough from composer for audit |
| `ref_unit` | str | passthrough from composer |

The Generator's role is **format verification**, not reasoning — it copies `filled_skeleton` verbatim if it conforms to the instruction's exact format, fixes only the format if not, and never invents content or replaces ref_values.

---

## Pipeline Versions

| Version | Anchor | Doc Refine | Composer | Generator | Runner |
|---|---|---|---|---|---|
| **v1** | `AnchorAgent` (block-based) | `DocRefineAgent` | `GlobalComposer` | `Generator` | [scripts/run_set1.py](scripts/run_set1.py) |
| **v2 (TOC)** | `AnchorAgentV2` (line-anchored hierarchical TOC) | `DocRefineAgentV2` (TOC sections) | `GlobalComposerV2` (records-based) | `Generator` (v1) | [scripts/run_set1_v2.py](scripts/run_set1_v2.py) |
| **v3** | `AnchorAgentV2` | `DocRefineAgentV2` | **`GlobalComposerV3`** (single-call 5-section reasoning) | **`GeneratorV2`** (filled_skeleton verifier) | [scripts/run_set1_v3.py](scripts/run_set1_v3.py) |

### What changed in each generation

- **v1 → v2**: char-offset block segmentation replaced by **line-anchored hierarchical TOC**. The LLM now emits L-numbers it can see verbatim in the prompt, and the agent code converts to char spans deterministically. This made `paper_L4 (citation chain)` go from **avg 69.5 → 89.5**.
- **v2 → v3**: composer is restructured to a **single LLM call that fills a 5-section reasoning protocol** (query_spec → doc_records → structure → completeness → filled_skeleton). The generator becomes a verifier rather than a re-reasoner. Key motivations:
  - Stop the composer from collapsing every task into a fixed `records[]` shape.
  - Make `ref_unit` (doc_id / entity_name / title / numeric / composite) explicit so the answer surface form is decided once and binding.
  - Make cross-doc graph form explicit (`fan_in`, `fan_out`, `chain`, `partition`, …) so doc-level membership cannot silently drop (e.g. fan-in citation chains).
  - Domain-blind by design: composer never sees `domain`/`language` flags, only the textual cues in question + instruction.

---

## Repository Layout

```
lambo/
├── README.md
├── lambo_v2/                            # Core Python package
│   ├── agents/
│   │   ├── anchor_agent.py              # v1: char-offset block segmentation
│   │   ├── anchor_agent_v2.py           # v2/v3: line-anchored hierarchical TOC
│   │   ├── doc_refine_agent.py          # v1: per-doc evidence loop (block-based)
│   │   ├── doc_refine_agent_v2.py       # v2/v3: per-doc evidence loop over TOC sections
│   │   ├── global_composer.py           # v1
│   │   ├── global_composer_v2.py        # v2: structured records[] composer
│   │   ├── global_composer_v3.py        # v3: single-call 5-section composer
│   │   ├── generator.py                 # v1/v2: projection_map-based generator
│   │   └── generator_v2.py              # v3: filled_skeleton verifier
│   ├── prompts/
│   │   ├── anchor/         {system,user}.txt   # v1
│   │   ├── doc_refine/     {system,user}.txt   # v1
│   │   ├── doc_refine_v2/  {system,user}.txt   # v2/v3 — TOC-aware loop
│   │   ├── compose/        {system,user}.txt   # v1
│   │   ├── compose_v2/     {system,user}.txt   # v2
│   │   ├── compose_v3/     {system,user}.txt   # v3 — 5-section protocol
│   │   ├── generate/       {system,user}.txt   # v1/v2
│   │   └── generate_v2/    {system,user}.txt   # v3 — filled_skeleton verifier
│   ├── eval/
│   │   ├── structured_eval.py
│   │   └── llm_judge.py
│   ├── scoring/
│   │   └── heuristic.py
│   ├── backend.py                       # QwenLocalClient / OpenAIClient (vLLM-compatible) / GeminiClient
│   ├── common.py
│   └── manifest.py
├── scripts/
│   ├── run_set1.py                      # v1 runner
│   ├── run_set1_v2.py                   # v2 runner (TOC)
│   ├── run_set1_v3.py                   # v3 runner (TOC + composer v3)
│   ├── indices_99.json                  # the 99-sample experiment manifest
│   └── loong_structured_reader.py
├── reference/
│   └── Loong/                           # Loong benchmark code & data (gitignored)
├── logs/                                # experiment outputs (gitignored)
└── lambo_prev/  dawon_all/  seohee_all/ # historical iterations preserved for reference
```

---

## Experiment Logs Layout

```
logs/<experiment_name>/
├── manifest.json                       # selected_index, sample_id, type, level, language
├── lambo_predictions.jsonl             # one JSON per sample with generate_response
├── samples/<sample_id>/
│   ├── anchors_v2.json                 # AnchorAgentV2 output
│   ├── DOC*_refine.json                # DocRefineAgentV2 evidence sheets
│   ├── composed_v2.json | composed_v3.json
│   ├── generator.json
│   └── error.json                      # only if the sample errored
└── reports/
    ├── structured_eval.json            # EM, pair-F1
    ├── llm_judge.json                  # 1-100 verdicts per sample + summary
    └── errors.json
```

---

## Reproducing Experiments

The 99-sample experiment uses the same indices across versions so v2/v3/v4/v5 are directly comparable. The manifest is in [scripts/indices_99.json](scripts/indices_99.json).

### Start the vLLM server (Qwen 3.5 27B with reasoning parser)

```bash
docker exec -d junyoungRAG_new bash -c "
  cd /workspace/lambo && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/bin/python -m vllm.entrypoints.openai.api_server \
    --model /workspace/lambo/models/Qwen3.5-27B \
    --served-model-name /workspace/lambo/models/Qwen3.5-27B \
    --tensor-parallel-size 4 \
    --max-model-len 50000 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 1 \
    --port 8000 \
    --trust-remote-code \
    --enforce-eager \
    --reasoning-parser qwen3 \
    > /workspace/lambo/logs/vllm_serve.log 2>&1
"
```

`--reasoning-parser qwen3` is required so vLLM exposes Qwen's thinking output via `reasoning_content`; the backend wraps it in `<think>…</think>` only for modules listed in `OPENAI_THINKING_MODULES`.

### Required `.env`

```
OPENAI_API_KEY=EMPTY
OPENAI_MODEL=/workspace/lambo/models/Qwen3.5-27B
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_DEFAULT_TEMPERATURE=0
OPENAI_THINKING_MODULES=doc_refine,doc_refine_v2
```

### Run the v3 pipeline (latest)

```bash
docker exec -d junyoungRAG_new bash -c "
  source /workspace/StructRAG/venv/bin/activate && \
  cd /workspace/lambo && \
  TQDM_DISABLE=1 python scripts/run_set1_v3.py \
    --backend openai \
    --selected_indices_path scripts/indices_99.json \
    --output_dir logs/lambo_v3_toc_99 \
    > logs/run_v3_toc_99.log 2>&1
"
```

### Run the v2 (TOC) pipeline

```bash
TQDM_DISABLE=1 python scripts/run_set1_v2.py \
  --backend openai \
  --selected_indices_path scripts/indices_99.json \
  --output_dir logs/lambo_v2_toc_99
```

### Run the v1 pipeline

```bash
python scripts/run_set1.py --force
```

### CLI Arguments (v2 / v3 runners)

| Argument | Default | Description |
|---|---|---|
| `--input_path` | `reference/Loong/data/loong_process.jsonl` | Loong dataset path |
| `--output_dir` | `logs/lambo_v3_toc_10` (v3) | per-experiment output dir |
| `--max_items` | `None` | limit number of samples |
| `--selected_indices` | `""` | comma-separated indices |
| `--selected_indices_path` | `""` | JSON file with `{"indices": [...]}` |
| `--max_refine_rounds` | `6` | DocRefineAgentV2 round cap |
| `--backend` | `openai` | `openai` (vLLM-compat) / `gemini` / `local` |
| `--force` | `False` | bypass per-sample caches |

---

## Environment

- **Model**: Qwen 3.5 27B Instruct served via vLLM OpenAI-compatible API at `localhost:8000`
- **Reasoning parser**: `qwen3` (so `<think>` content survives into the trace)
- **GPUs**: 4 × NVIDIA RTX 3090 (24 GB), TP=4, `enforce_eager`, `max_num_seqs=1`
- **Container**: `junyoungRAG_new` (Docker)
- **Python venv**: `/workspace/StructRAG/venv` (experiment scripts) — vLLM itself runs from `/opt/conda/bin/python`
- **Token budget**: 50 000 input / 8 192 output for composer/generator; 50 000 model length

---

## Results Summary

99-sample fixed manifest ([scripts/indices_99.json](scripts/indices_99.json)), Qwen 3.5 27B, LLM-as-judge 1-100.

| Pipeline | n done | Avg score | Perfect | Notes |
|---|---|---|---|---|
| `lambo_v2_exper99_qwen35_27b` (v2 — char-anchored) | 99 | **74.14** | 59 / 99 | original v2 baseline |
| `lambo_v4_qwen35_27b_balanced99` | 99 | **80.28** | 64 / 99 | balanced multi-experiment baseline |
| `lambo_v2_toc_99` (v2 — line-anchored TOC) | 92 | 73.04 | 59 / 92 | 7 timeouts on long legal docs |
| `lambo_v3_toc_99` | _interrupted_ | — | — | composer v3 launched, halted for design iteration |

Per type/level (v4 vs v2-toc):

| Type / Level | v4 avg | v2-toc avg | Δ |
|---|---|---|---|
| paper_L3 | 87.9 | 72.9 | −15.0 |
| **paper_L4 (citation chain)** | 71.6 | **89.5** | **+17.9** |
| legal_L3 | 66.0 | 50.0 | −16.0 (+ 6 timeouts) |
| legal_L4 | 100.0 | 100.0 | 0 |
| financial_L1 | 91.7 | 83.3 | −8.4 |
| financial_L2 | 94.2 | 75.0 | −19.2 |

The TOC line-anchored design dramatically helps citation-chain tasks (paper_L4) but produces evidence that is too rich for short single-fact tasks (financial L1/L2, paper_L3) and overflows the 50K window on long legal documents (legal_L3 timeouts). The v3 composer is a response to the answer-shape mismatches observed in these regressions.

---

## Key Design Decisions

- **Line-numbered TOC anchoring** ([lambo_v2/agents/anchor_agent_v2.py](lambo_v2/agents/anchor_agent_v2.py)). The LLM only emits L-numbers it can see verbatim; char spans are computed deterministically. This avoids the unreliable char-offset estimation of v1.
- **Domain-blind composer (v3)**. The composer never sees `domain` or `language` — answer-surface decisions (`ref_unit`) come from textual cues in the instruction example or question wording.
- **Single LLM call with explicit reasoning protocol**. `compose_v3/system.txt` enforces a 5-section JSON output in fixed order, so each reasoning stage is auditable separately.
- **Generator as verifier, not re-reasoner**. `generator_v2` copies `filled_skeleton` if it conforms; otherwise reformats only, never edits ref_values.
- **Per-module thinking control**. `OPENAI_THINKING_MODULES` env var enables `<think>` for selected agents only (default: `doc_refine`, `doc_refine_v2`).
- **Defensive post-processing**. `_normalize_and_verify` in `global_composer_v3.py` auto-fills doc records the LLM omitted, validates `selected_count`, and emits `warnings[]` for inconsistencies.
- **No leakage of dataset metadata to the LLM**: `type`, `level`, and `set` are used only for evaluation/manifest purposes, not in any prompt.
