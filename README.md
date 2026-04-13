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
| Backend | `lambo/backend.py` | Qwen2.5-32B-Instruct local inference |

## Directory Structure

```
lambo/
├── lambo/                  # Main Python package
│   ├── agents/             # LLM-calling agent modules
│   ├── scoring/            # Heuristic scoring (non-LLM)
│   ├── eval/               # Evaluation modules
│   ├── prompts/            # Prompt templates by role
│   │   ├── anchor/
│   │   ├── doc_refine/
│   │   ├── compose/
│   │   └── generate/
│   ├── backend.py          # QwenLocalClient
│   ├── common.py           # Shared utilities
│   └── manifest.py         # Sample selection and manifest
├── scripts/                # Entry points
│   └── run_set1.py         # Main experiment runner
├── reference/              # External reference code
│   ├── Agentic-R/
│   └── Loong/
├── logs/                   # Experiment outputs
└── README.md
```

## Quick Start

```bash
docker exec junyoungRAG bash -lc '
  cd /workspace/lambo &&
  source /workspace/StructRAG/venv/bin/activate &&
  python scripts/run_set1.py --max_items 1 --force
'
```

Full 10-sample run:
```bash
docker exec junyoungRAG bash -lc '
  cd /workspace/lambo &&
  source /workspace/StructRAG/venv/bin/activate &&
  python scripts/run_set1.py --force
'
```

## Key Design Decisions

- **AutoRefine prompt accumulation**: Each round's trace (<think>/<search>/<info>/<refine>) is appended to the next round's prompt. Old <info> blocks are compressed after 2 rounds to save tokens.
- **Heuristic pre-ranking**: Anchors are scored by domain priors (section type, query overlap, quoted terms) before LLM selection — the LLM only sees the top-ranked shortlist.
- **Typed evidence**: Each evidence item has an `item_type` (direct_answer, comparison_value, classification_feature, relation_edge, negative_evidence) and `owner_entity` for downstream entity resolution.
- **No cheating**: `type`, `level`, `task_mode` are used only in heuristic scoring internals, never passed to LLM prompts.
- **Negative evidence**: Documents with no relevant evidence are explicitly tracked (scan_result="no_evidence"), not silently dropped.
