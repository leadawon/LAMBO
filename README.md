# LAMBO — Long-document Agentic Multi-doc Benchmark Optimizer

AutoRefine-style agentic RAG pipeline for the [Loong benchmark](https://github.com/Loong-Long-Document-QA).

## Pipeline Flow

```
┌─────────────────┐
│  Anchor Agent    │  Per-doc: segment → anchor tiling → doc_map
└────────┬────────┘
         ▼
┌─────────────────┐
│ Doc Refine Agent │  Per-doc iterative loop:
│  (AutoRefine)    │    think → search → info → refine (repeat)
│                  │  Output: typed evidence sheet per document
└────────┬────────┘
         ▼
┌─────────────────┐
│ Global Composer  │  Cross-doc: entity resolution (projection_map),
│                  │  comparison, relation extraction
└────────┬────────┘
         ▼
┌─────────────────┐
│   Generator      │  DOC id → entity name projection,
│                  │  instruction-shape serialization
└────────┬────────┘
         ▼
┌─────────────────┐
│   Evaluation     │  structured_eval (EM, F1) + LLM judge (1-100)
└─────────────────┘
```

## Components

| Module | Path | Role |
|--------|------|------|
| Anchor Agent | `lambo/agents/anchor_agent.py` | Segments each document into semantic anchors with summaries |
| Doc Refine Agent | `lambo/agents/doc_refine_agent.py` | Per-doc AutoRefine loop extracting typed evidence items |
| Global Composer | `lambo/agents/global_composer.py` | Cross-doc entity resolution + relation synthesis |
| Generator | `lambo/agents/generator.py` | Final answer with DOC→entity projection |
| Heuristic Scoring | `lambo/scoring/heuristic.py` | Domain-aware anchor ranking (no LLM) |
| LLM Judge | `lambo/eval/llm_judge.py` | Loong-style 1-100 scoring |
| Structured Eval | `lambo/eval/structured_eval.py` | Exact match + pair-F1 metrics |
| Backend | `lambo/backend.py` | Hugging Face Qwen Transformers backend (default: `Qwen/Qwen2.5-7B-Instruct`) |

## Directory Structure

```
lambo/                              # Repository root
├── lambo/                          # Core Python package (v2 — latest)
│   ├── agents/                     # LLM-calling agent modules
│   │   ├── anchor_agent.py         #   Stage 1: document segmentation & anchor tiling
│   │   ├── doc_refine_agent.py     #   Stage 2: per-doc iterative evidence extraction
│   │   ├── global_composer.py      #   Stage 3: cross-doc entity resolution & synthesis
│   │   └── generator.py            #   Stage 4: final answer generation
│   ├── scoring/
│   │   └── heuristic.py            # Domain-aware anchor ranking (no LLM)
│   ├── eval/
│   │   ├── structured_eval.py      # Exact match + pair-F1 metrics
│   │   └── llm_judge.py            # Loong-style 1-100 LLM scoring
│   ├── prompts/                    # Prompt templates per stage
│   │   ├── anchor/                 #   system.txt, user.txt
│   │   ├── doc_refine/             #   system.txt, user.txt
│   │   ├── compose/                #   system.txt, user.txt
│   │   └── generate/               #   system.txt, user.txt
│   ├── backend.py                  # QwenLocalClient (Transformers backend)
│   ├── common.py                   # Shared utilities
│   └── manifest.py                 # Sample selection & manifest builder
│
├── scripts/
│   └── run_set1.py                 # Main experiment entry point (v2)
│
├── dawon/                          # v1 runner — original dawon experiment code
│   ├── anchor/                     #   v1 agent modules & prompts
│   ├── data/                       #   v1 dataset manifests
│   ├── logs/                       #   v1 experiment logs
│   └── run_set1_10.py              #   v1 entry point
│
├── dawonv2/                        # v2 iteration — intermediate experiment code
│   ├── anchor/                     #   v2 agent modules & prompts
│   └── data/                       #   v2 dataset manifests
│
├── dawonv3/                        # v3 iteration — further refinements
│   ├── anchor/                     #   v3 agent modules & prompts
│   ├── data/                       #   v3 dataset manifests
│   └── logs/                       #   v3 experiment logs
│
├── logs/                           # Experiment output logs (see below)
├── reference/                      # External reference code (excluded from git)
│   ├── Agentic-R/                  #   Agentic-R baseline reference
│   └── Loong/                      #   Loong benchmark data & evaluation code
│
├── script/                         # Legacy script directory
│   └── anchor/                     #   Original anchor-based pipeline scripts
│
├── .gitignore
└── README.md
```

## Experiment Logs

All experiment outputs are stored under `logs/`. Each experiment directory follows this structure:

```
logs/<experiment_name>/
├── manifest.json                   # Selected sample indices and metadata
├── lambo_predictions.jsonl         # Final predictions (one JSON per line)
├── samples/                        # Per-sample intermediate outputs
│   └── <sample_id>/
│       ├── anchors.json            # Stage 1: anchor extraction result
│       ├── DOC*_refine.json        # Stage 2: per-doc evidence sheets
│       ├── composed.json           # Stage 3: cross-doc composition
│       └── generator.json          # Stage 4: final answer
└── reports/
    ├── structured_eval.json        # EM / pair-F1 metrics
    ├── llm_judge.json              # LLM judge 1-100 scores per sample
    └── errors.json                 # Error tracking
```

### Experiment History

| Experiment | Description | Samples | EM | Pair-F1 | LLM Judge Avg |
|------------|-------------|---------|-----|---------|---------------|
| `lambo_set1_10_smoke_doccall` | Smoke test (doc-call backend) | 1 | 1.00 | — | — |
| `lambo_agentic_smoke` | Smoke test (agentic pipeline) | 1 | 0.00 | — | — |
| `lambo_set1_10_doccall` | Full run with doc-call backend | 10 | 0.10 | 0.313 | — |
| `lambo_agentic_set1_10` | Full run with agentic pipeline | 10 | 0.10 | 0.365 | 27.0 |
| **`lambo_v2_set1_10`** | **Latest v2 pipeline (current)** | **10** | **0.20** | **0.475** | **53.5** |

### Latest Results: `lambo_v2_set1_10` (2026-04-13)

Per-sample breakdown:

| Sample | Type | Level | LLM Score | EM | Notes |
|--------|------|-------|-----------|-----|-------|
| `financial_level1_914` | financial | 1 | 100 | O | Perfect answer |
| `legal_level1_725` | legal | 1 | 95 | X | Minor formatting difference |
| `financial_level2_1065` | financial | 2 | 20 | X | Wrong company identified |
| `legal_level2_801` | legal | 2 | 0 | X | No evidence found in any document |
| `financial_level3_1280` | financial | 3 | 60 | X | Partial classification error |
| `legal_level3_469` | legal | 3 | 80 | X | 2 documents swapped between categories |
| `paper_level3_63` | paper | 3 | 75 | X | Reference missed, citation correct |
| `financial_level4_1504` | financial | 4 | 85 | X | Minor numerical difference |
| `legal_level4_596` | legal | 4 | 20 | O | EM correct but judge scored low |
| `paper_level4_32` | paper | 4 | 0 | X | No evidence extracted |

Performance by level:

| Level | Avg LLM Score | Description |
|-------|---------------|-------------|
| Level 1 | 97.5 | Single-doc, single-fact retrieval |
| Level 2 | 10.0 | Multi-doc comparison |
| Level 3 | 71.7 | Multi-doc classification / relation |
| Level 4 | 35.0 | Complex cross-doc reasoning |

### Version Evolution

- **dawon (v1)**: Initial implementation using a vLLM OpenAI-compatible server backend. Basic anchor + search pipeline with separate judge evaluation.
- **dawonv2**: Intermediate iteration with refined anchor agent and prompts.
- **dawonv3**: Further refinements to evidence extraction and prompt engineering.
- **lambo (v2 — current)**: Complete rewrite with 4-stage agentic pipeline (AnchorAgent → DocRefineAgent → GlobalComposer → Generator). Uses Transformers backend directly. Added heuristic pre-ranking, typed evidence, and cross-document entity resolution.

## Quick Start

```bash
# Smoke test (1 sample)
docker exec junyoungRAG bash -lc '
  cd /workspace/lambo &&
  source /workspace/StructRAG/venv/bin/activate &&
  python scripts/run_set1.py --max_items 1 --force
'

# Full 10-sample run
docker exec junyoungRAG bash -lc '
  cd /workspace/lambo &&
  source /workspace/StructRAG/venv/bin/activate &&
  python scripts/run_set1.py --force
'
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_path` | `reference/Loong/data/loong_process.jsonl` | Input dataset path |
| `--output_dir` | `logs/lambo_v2_set1_10` | Output directory |
| `--manifest_mode` | `set1_10` | `set1_10`, `selected_indices`, or `input_order` |
| `--selected_indices` | `` | Comma/space-separated original record indices |
| `--selected_indices_path` | `` | JSON/text file containing selected original record indices |
| `--max_items` | `None` | Limit number of samples after manifest selection |
| `--force` | `False` | Bypass cache and recompute all stages |
| `--max_search_rounds` | `None` | Alias for `--max_refine_rounds`; overrides the per-document search/refine round cap when set |

### Running 10 vs 99 Samples

Fixed 10-sample set1 run:

```bash
cd /workspace/rag/LAMBO
PYTHONPATH=/workspace/rag/LAMBO \
LAMBO_MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
python scripts/run_set1.py \
  --input_path /workspace/rag/LAMBO_before/Loong/data/loong_process.jsonl \
  --manifest_mode set1_10 \
  --output_dir /workspace/rag/LAMBO/logs/lambo_v2_set1_10_multiturn
```

Balanced 99-sample run using a saved indices file:

```bash
cd /workspace/rag/LAMBO
PYTHONPATH=/workspace/rag/LAMBO \
LAMBO_MODEL_ID=Qwen/Qwen2.5-7B-Instruct \
python scripts/run_set1.py \
  --input_path /workspace/rag/LAMBO_before/Loong/data/loong_process.jsonl \
  --manifest_mode selected_indices \
  --selected_indices_path /workspace/rag/LAMBO/dawonv5/data/loong_set1_balanced99_indices.json \
  --output_dir /workspace/rag/LAMBO/logs/lambo_v2_set1_99_multiturn
```

You can also pass indices directly:

```bash
python scripts/run_set1.py \
  --manifest_mode selected_indices \
  --selected_indices 914,725,1065
```

## Environment

- **Model**: `Qwen/Qwen2.5-7B-Instruct` via Hugging Face Transformers by default (optional local path override with `LAMBO_MODEL_DIR`)
- **GPU**: 4x NVIDIA RTX 3090 (24GB each)
- **Container**: `junyoungRAG` Docker container
- **Python env**: `/workspace/StructRAG/venv`
- **Dataset**: Loong benchmark (1,600 records, 10 selected for set-1)

## Key Design Decisions

- **AutoRefine prompt accumulation**: Each round's trace (`<think>`/`<search>`/`<info>`/`<refine>`) is appended to the next round's prompt. Old `<info>` blocks are compressed after 2 rounds to save tokens.
- **Heuristic pre-ranking**: Anchors are scored by domain priors (section type, query overlap, quoted terms) before LLM selection — the LLM only sees the top-ranked shortlist.
- **Typed evidence**: Each evidence item has an `item_type` (direct_answer, comparison_value, classification_feature, relation_edge, negative_evidence) and `owner_entity` for downstream entity resolution.
- **No cheating**: `type`, `level`, `task_mode` are used only in heuristic scoring internals, never passed to LLM prompts.
- **v2 multi-turn document refinement**: `lambo/agents/doc_refine_agent.py` now runs as a stateful multi-turn searcher rather than a one-shot anchor opener. It reranks anchors across rounds, can continue locally around the current anchor, and can rewrite the working query when the current evidence is insufficient.
- **`SearchState` as agent memory**: In the v2 pipeline, `SearchState` acts as the per-document search memory. Its main fields are `current_query`, `must_find`, `seen_anchor_ids`, `known_items`, `missing_slots`, `last_action`, and local/global frontier anchor ids. This lets the agent remember which anchors it already opened, what evidence it has already observed, what information is still missing, and whether the next move should stay local or jump elsewhere in the document.
- **Summary is routing, not evidence**: Anchor summaries in v2 are now treated as navigation hints only. Search may use them to choose which anchor to open next, but downstream refinement must verify evidence against the opened anchor's raw text instead of trusting the summary.
- **Raw-text evidence validation**: The v2 doc refine agent now accepts only `<answer>` content that is supported by previously opened raw anchor text. This is meant to stop the model from filling evidence directly from anchor summaries or from search notes.
- **Negative evidence**: Documents with no relevant evidence are explicitly tracked (`scan_result="no_evidence"`), not silently dropped.
