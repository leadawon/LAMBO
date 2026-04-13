"""Run the LAMBO v2 pipeline on set-1 Loong samples.

Pipeline: AnchorAgent → DocRefineAgent (per-doc) → GlobalComposer → Generator
Evaluation: structured_eval (EM, F1) + LLM judge (1-100)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lambo.agents.anchor_agent import AnchorAgent
from lambo.agents.doc_refine_agent import DocRefineAgent
from lambo.agents.global_composer import GlobalComposer
from lambo.agents.generator import Generator
from lambo.backend import get_default_client
from lambo.common import (
    current_query_from_record,
    ensure_dir,
    instruction_from_record,
    json_dumps_pretty,
    safe_filename,
    write_json,
    write_jsonl,
)
from lambo.eval.structured_eval import evaluate_predictions
from lambo.eval.llm_judge import run_llm_judge
from lambo.manifest import build_set1_manifest, load_records, save_manifest


DEFAULT_INPUT_PATH = PROJECT_ROOT / "reference" / "Loong" / "data" / "loong_process.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "lambo_v2_set1_10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LAMBO v2 AutoRefine pipeline on set1 samples.")
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_refine_rounds", type=int, default=6)
    return parser.parse_args()


def ensure_layout(output_dir: Path) -> Dict[str, Path]:
    layout = {
        "root": output_dir,
        "samples": output_dir / "samples",
        "reports": output_dir / "reports",
    }
    for p in layout.values():
        ensure_dir(p)
    return layout


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    layout = ensure_layout(output_dir)

    records = load_records(input_path)
    manifest = build_set1_manifest(records)
    if args.max_items is not None:
        manifest = manifest[: args.max_items]
    save_manifest(manifest, layout["root"] / "manifest.json")

    llm = get_default_client()

    anchor_agent = AnchorAgent(llm=llm)
    doc_refine_agent = DocRefineAgent(
        llm=llm,
        max_rounds=args.max_refine_rounds,
    )
    composer = GlobalComposer(llm=llm)
    generator = Generator(llm=llm)

    prediction_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for item in manifest:
        record = records[item.selected_index]
        sample_dir = layout["samples"] / safe_filename(item.sample_id)
        ensure_dir(sample_dir)
        print(f"\n==== sample {item.sample_id} (idx={item.selected_index}) ====", flush=True)

        try:
            question = current_query_from_record(
                str(record.get("question", "")).strip(),
                str(record.get("instruction", "")).strip(),
            )
            instruction = instruction_from_record(
                str(record.get("instruction", "")).strip(),
            )

            # Stage 1: Anchor Agent
            anchor_payload = anchor_agent.run(
                record=record, sample_dir=sample_dir, force=args.force,
            )

            # Stage 2: DocRefineAgent per document
            doc_sheets: List[Dict[str, Any]] = []
            for doc_payload in anchor_payload["docs"]:
                print(
                    f"  doc {doc_payload['doc_id']} '{doc_payload['doc_title'][:40]}' "
                    f"anchors={doc_payload['anchor_count']}",
                    flush=True,
                )
                # Dynamic max_rounds: no more rounds than anchors, capped by user arg
                doc_refine_agent.max_rounds = min(
                    args.max_refine_rounds,
                    max(1, doc_payload["anchor_count"]),
                )
                sheet = doc_refine_agent.run(
                    question=question,
                    instruction=instruction,
                    record=record,
                    doc_payload=doc_payload,
                    sample_dir=sample_dir,
                    force=args.force,
                )
                doc_sheets.append(sheet)
                evidence_preview = sheet.get("evidence", "")[:80]
                print(
                    f"    -> {sheet['scan_result']} "
                    f"| rounds={sheet.get('rounds_used',0)} "
                    f"| evidence={evidence_preview!r}",
                    flush=True,
                )

            # Stage 3: Global Composer
            composed = composer.run(
                question=question,
                instruction=instruction,
                doc_sheets=doc_sheets,
                sample_dir=sample_dir,
                force=args.force,
            )
            print(
                f"  composer -> projection_map keys={list(composed.get('projection_map',{}).keys())} "
                f"| records={len(composed.get('records',[]))}",
                flush=True,
            )

            # Stage 4: Generator
            gen_out = generator.run(
                question=question,
                instruction=instruction,
                composed=composed,
                sample_dir=sample_dir,
                force=args.force,
            )
            final_answer = gen_out["final_answer"]

            keep = ("id", "type", "level", "question", "instruction", "answer", "answer_topology")
            pred_row = {k: record.get(k) for k in keep if k in record}
            pred_row["selected_index"] = item.selected_index
            pred_row["sample_id"] = item.sample_id
            if isinstance(final_answer, (dict, list)):
                pred_row["generate_response"] = json.dumps(final_answer, ensure_ascii=False)
            else:
                pred_row["generate_response"] = str(final_answer)
            pred_row["lambo_trace_dir"] = str(sample_dir)
            prediction_rows.append(pred_row)
            print(f"  -> answer: {str(final_answer)[:200]}", flush=True)

        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"  !! error on {item.sample_id}: {exc}\n{tb}", flush=True)
            errors.append({"sample_id": item.sample_id, "error": str(exc), "traceback": tb})
            write_json(sample_dir / "error.json", errors[-1])

    prediction_path = write_jsonl(layout["root"] / "lambo_predictions.jsonl", prediction_rows)

    # Structured evaluation
    try:
        structured_summary = evaluate_predictions(
            [
                {
                    **row,
                    "generate_response": (
                        json.loads(row["generate_response"])
                        if isinstance(row["generate_response"], str)
                        and row["generate_response"].startswith(("{", "["))
                        else row["generate_response"]
                    ),
                }
                for row in prediction_rows
            ]
        )
    except Exception as exc:  # noqa: BLE001
        structured_summary = {"error": str(exc)}
    write_json(layout["reports"] / "structured_eval.json", structured_summary)
    write_json(layout["reports"] / "errors.json", {"count": len(errors), "errors": errors})

    # LLM judge
    try:
        judge_out = run_llm_judge(llm=llm, prediction_rows=prediction_rows)
    except Exception as exc:  # noqa: BLE001
        judge_out = {"summary": {"error": str(exc)}, "verdicts": []}
    write_json(layout["reports"] / "llm_judge.json", judge_out)

    print(
        json_dumps_pretty(
            {
                "prediction_path": str(prediction_path),
                "structured_summary": {
                    k: v
                    for k, v in (structured_summary or {}).items()
                    if k != "per_sample"
                },
                "llm_judge_summary": judge_out.get("summary", {}),
                "num_errors": len(errors),
            }
        )
    )


if __name__ == "__main__":
    main()
