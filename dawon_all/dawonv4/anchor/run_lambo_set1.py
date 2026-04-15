"""Run the redesigned LAMBO pipeline on 10 set-1 Loong samples."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


DAWON_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DAWON_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dawonv4.anchor.anchor_agent import AnchorAgent
from dawonv4.anchor.answer_writer import AnswerWriter
from dawonv4.anchor.backend import get_default_client
from dawonv4.anchor.common import (
    current_query_from_record,
    ensure_dir,
    instruction_from_record,
    json_dumps_pretty,
    safe_filename,
    write_json,
    write_jsonl,
)
from dawonv4.anchor.evaluate_structured import evaluate_predictions
from dawonv4.anchor.llm_judge import run_llm_judge
from dawonv4.anchor.manifest import (
    build_manifest_for_indices,
    build_set1_manifest,
    load_records,
    save_manifest,
)
from dawonv4.anchor.paths import resolve_loong_process_path
from dawonv4.anchor.relation_refiner import RelationRefiner
from dawonv4.anchor.search_agent import ExtractAgent, SearchAgent


DEFAULT_INPUT_PATH = resolve_loong_process_path(strict=False)
DEFAULT_OUTPUT_DIR = DAWON_ROOT / "logs" / "lambo_agentic_set1_10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run redesigned agentic LAMBO pipeline on 10 set1 samples.")
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH or ""))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_search_rounds", type=int, default=5)
    parser.add_argument("--selected_indices", type=str, default="")
    parser.add_argument("--selected_indices_path", type=str, default="")
    return parser.parse_args()


def parse_index_tokens(text: str) -> List[int]:
    return [int(token) for token in text.replace(",", " ").split()]


def load_selected_indices(args: argparse.Namespace) -> List[int] | None:
    indices: List[int] = []
    if args.selected_indices_path:
        path = Path(args.selected_indices_path)
        text = path.read_text(encoding="utf-8").strip()
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                indices.extend(parse_index_tokens(text))
            else:
                if isinstance(payload, dict):
                    payload = payload.get("indices", [])
                if not isinstance(payload, list):
                    raise ValueError(f"selected_indices_path must contain a JSON list or dict.indices: {path}")
                indices.extend(int(value) for value in payload)

    if args.selected_indices:
        indices.extend(parse_index_tokens(args.selected_indices))

    if not indices:
        return None
    return indices


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
    input_path = Path(args.input_path) if args.input_path else resolve_loong_process_path(strict=True)
    output_dir = Path(args.output_dir)
    layout = ensure_layout(output_dir)

    records = load_records(input_path)
    selected_indices = load_selected_indices(args)
    if selected_indices is None:
        manifest = build_set1_manifest(records)
    else:
        manifest = build_manifest_for_indices(records, selected_indices)
    if args.max_items is not None:
        manifest = manifest[: args.max_items]
    save_manifest(manifest, layout["root"] / "manifest.json")

    llm = get_default_client()
    prompt_dir = Path(__file__).resolve().parent / "prompts"
    anchor_agent = AnchorAgent(llm=llm, prompt_dir=prompt_dir)
    extract_agent = ExtractAgent(llm=llm, prompt_dir=prompt_dir)
    search_agent = SearchAgent(
        llm=llm,
        extract_agent=extract_agent,
        prompt_dir=prompt_dir,
        max_rounds=args.max_search_rounds,
    )
    refiner = RelationRefiner(llm=llm, prompt_dir=prompt_dir)
    writer = AnswerWriter(llm=llm, prompt_dir=prompt_dir)

    prediction_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for item in manifest:
        record = records[item.selected_index]
        sample_dir = layout["samples"] / safe_filename(item.sample_id)
        ensure_dir(sample_dir)
        print(f"\n==== sample {item.sample_id} (idx={item.selected_index}) ====", flush=True)
        try:
            query = current_query_from_record(
                str(record.get("question", "")).strip(),
                str(record.get("instruction", "")).strip(),
            )
            instruction = instruction_from_record(
                str(record.get("instruction", "")).strip()
            )

            anchor_payload = anchor_agent.run(
                record=record, sample_dir=sample_dir, force=args.force
            )

            doc_results: List[Dict[str, Any]] = []
            for doc_payload in anchor_payload["docs"]:
                print(
                    f"  doc {doc_payload['doc_id']} '{doc_payload['doc_title'][:40]}' "
                    f"anchors={doc_payload['anchor_count']}",
                    flush=True,
                )
                search_agent.max_rounds = max(
                    args.max_search_rounds,
                    min(8, doc_payload["anchor_count"]),
                )
                search_out = search_agent.run(
                    query=query,
                    instruction=instruction,
                    doc_payload=doc_payload,
                    sample_dir=sample_dir,
                    force=args.force,
                )
                doc_results.append(search_out)

            refine_out = refiner.run(
                query=query,
                instruction=instruction,
                doc_results=doc_results,
                sample_dir=sample_dir,
                force=args.force,
            )
            answer_out = writer.run(
                query=query,
                instruction=instruction,
                relations=refine_out["relations"],
                sample_dir=sample_dir,
                force=args.force,
            )
            final_answer = answer_out["final_answer"]

            keep = ("id","type","level","question","instruction","answer","answer_topology")
            pred_row = {k: record.get(k) for k in keep if k in record}
            pred_row["selected_index"] = item.selected_index
            pred_row["sample_id"] = item.sample_id
            # For evaluate_structured/json compatibility and for Loong judge.
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

    # Structured evaluation (exact / pair-F1 against gold `answer`).
    try:
        structured_summary = evaluate_predictions(
            [
                {
                    **row,
                    # evaluate_predictions reads row["generate_response"] as the prediction
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

    # LLM judge (Loong-style) — reuses the same local Qwen client.
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
