# LAMBO Pipeline Visualizer

A Streamlit web UI that turns any sample under `logs/<experiment>/samples/` into a MODORA-style document-browsing experience:

- **Document Browser** — each document is reconstructed under its hierarchical TOC, with section text placed under the anchor's section heading. Sections that DocRefineAgent actually *opened* are flagged (purple bar + "opened" badge), and evidence spans returned by the agent are highlighted inline (yellow `<mark>`) so you can immediately see "where the answer came from". Evidence is collected from both `refine.evidence` and the trace's `<answer>...</answer>` payload.
- **Search Trace** — the per-doc `<think> / <search> / <info> / <answer>` log, color-coded. The same evidence spans are highlighted inside matching `<info>` blocks.
- **Composer / Generator** — query_spec (v3), projection map, records, filled skeleton, final answer, and the LLM judge rationale.

The page picks up any experiment dir in `logs/`, but defaults to `logs/lambo_v2_toc_99`.

## Run

Inside the project's existing StructRAG venv (Streamlit was installed there):

```bash
docker exec -it junyoungRAG_new bash -c "
  source /workspace/StructRAG/venv/bin/activate && \
  cd /workspace/lambo && \
  streamlit run webapp/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true
"
```

Then open <http://localhost:8501> (or <http://&lt;host-ip&gt;:8501>) in your browser. If you're using port forwarding from a remote machine, forward 8501.

## Pick a different experiment

Use the sidebar's "Experiment log directory" field — point it to any `logs/<experiment_name>/` containing `samples/`, `manifest.json`, `lambo_predictions.jsonl` and `reports/`. The app supports both v2 (`composed_v2.json`) and v3 (`composed_v3.json`) outputs.

## Sidebar toggles

- **Show all sections (not just opened)** — turn off to collapse the document view down to only the sections the agent actually opened.
- **Expand section text** — turn off to show only TOC headings (good for very long documents).
