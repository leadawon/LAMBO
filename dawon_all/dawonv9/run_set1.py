"""Run the dawonv9 pipeline on set-1 Loong samples.

Same 4-stage flow as lambo v2 (AnchorAgent → DocRefineAgent → GlobalComposer
→ Generator) but wraps three of the stages with v5/v7 enrichment.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


# Make both the `lambo` package and the `dawonv9` package importable.
LAMBO_ROOT = Path(__file__).resolve().parents[2]
if str(LAMBO_ROOT) not in sys.path:
    sys.path.insert(0, str(LAMBO_ROOT))
DAWON_ROOT = Path(__file__).resolve().parents[1]
if str(DAWON_ROOT) not in sys.path:
    sys.path.insert(0, str(DAWON_ROOT))

from lambo.agents.generator import Generator
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
from lambo.manifest import (
    ManifestItem,
    build_set1_manifest,
    load_records,
    save_manifest,
)

from dawonv9.agents import (
    CrossDocGlobalComposer,
    EnrichedAnchorAgent,
    EnrichedDocRefineAgent,
)
from dawonv9.backend import get_default_client


DEFAULT_INPUT_PATH = LAMBO_ROOT / "reference" / "Loong" / "data" / "loong_process.jsonl"
DEFAULT_OUTPUT_DIR = DAWON_ROOT / "logs" / "dawonv9_set1_10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dawonv9 (lambo v2 + v5 enrichment) on set1 samples."
    )
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_refine_rounds", type=int, default=6)
    parser.add_argument(
        "--selected_indices",
        type=str,
        default="",
        help="Comma/space separated indices into the full Loong JSONL.",
    )
    parser.add_argument(
        "--selected_indices_path",
        type=str,
        default="",
        help="JSON file containing a list or {\"indices\": [...]} of row indices.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="local",
        choices=["local", "gemini"],
        help="LLM backend: 'local' Qwen2.5-32B or 'gemini' API.",
    )
    return parser.parse_args()


def _parse_index_tokens(text: str) -> List[int]:
    return [int(token) for token in text.replace(",", " ").split()]


def load_selected_indices(args: argparse.Namespace) -> List[int] | None:
    indices: List[int] = []
    if args.selected_indices_path:
        text = Path(args.selected_indices_path).read_text(encoding="utf-8").strip()
        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                indices.extend(_parse_index_tokens(text))
            else:
                if isinstance(payload, dict):
                    payload = payload.get("indices", [])
                if not isinstance(payload, list):
                    raise ValueError(
                        f"selected_indices_path must contain a JSON list or "
                        f"dict.indices: {args.selected_indices_path}"
                    )
                indices.extend(int(v) for v in payload)
    if args.selected_indices:
        indices.extend(_parse_index_tokens(args.selected_indices))
    return indices or None


def _item_for_index(record: Dict[str, Any], index: int) -> ManifestItem:
    return ManifestItem(
        selected_index=index,
        record_id=str(record.get("id", "unknown")),
        set_id=int(record.get("set", 0) or 0),
        record_type=str(record.get("type", "unknown")),
        level=int(record.get("level", 0) or 0),
        language=str(record.get("language", "unknown")),
        question=str(record.get("question", "")).strip(),
        sample_id=f"{record.get('type', 'unknown')}_level{record.get('level', 0)}_{index}",
    )


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

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.backend == "gemini":
        output_dir = DAWON_ROOT / "logs" / "dawonv9_set1_10_gemini"
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    layout = ensure_layout(output_dir)

    records = load_records(input_path)
    selected_indices = load_selected_indices(args)
    if selected_indices is None:
        manifest = build_set1_manifest(records)
    else:
        manifest = [_item_for_index(records[i], i) for i in selected_indices]
    if args.max_items is not None:
        manifest = manifest[: args.max_items]
    save_manifest(manifest, layout["root"] / "manifest.json")

    print(f"Backend: {args.backend}", flush=True)
    llm = get_default_client(backend=args.backend)

    anchor_agent = EnrichedAnchorAgent(llm=llm)
    doc_refine_agent = EnrichedDocRefineAgent(llm=llm, max_rounds=args.max_refine_rounds)
    composer = CrossDocGlobalComposer(llm=llm)
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
                str(record.get("instruction", "")).strip()
            )

            anchor_payload = anchor_agent.run(
                record=record, sample_dir=sample_dir, force=args.force
            )

            doc_sheets: List[Dict[str, Any]] = []
            for doc_payload in anchor_payload["docs"]:
                print(
                    f"  doc {doc_payload['doc_id']} '{doc_payload['doc_title'][:40]}' "
                    f"anchors={doc_payload['anchor_count']}",
                    flush=True,
                )
                doc_refine_agent.max_rounds = min(
                    args.max_refine_rounds,
                    max(1, doc_payload["anchor_count"]),
                )
                sheet = doc_refine_agent.run(
                    question=question,
                    instruction=instruction,
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

            composed = composer.run(
                question=question,
                instruction=instruction,
                doc_sheets=doc_sheets,
                sample_dir=sample_dir,
                force=args.force,
            )
            print(
                f"  composer -> projection_map keys={list(composed.get('projection_map',{}).keys())} "
                f"| records={len(composed.get('records',[]))} "
                f"| cross_doc_matches="
                f"{len((composed.get('cross_doc_citations') or {}).get('matches', []))}",
                flush=True,
            )

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

    prediction_path = write_jsonl(
        layout["root"] / "lambo_predictions.jsonl", prediction_rows
    )

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

    try:
        judge_out = run_llm_judge(llm=llm, prediction_rows=prediction_rows)
    except Exception as exc:  # noqa: BLE001
        judge_out = {"summary": {"error": str(exc)}, "verdicts": []}
    write_json(layout["reports"] / "llm_judge.json", judge_out)

    # Consolidated score files so `bash run_exper99.sh` leaves
    # metrics.json / scores.json / summary.txt behind for later review.
    structured_public = {
        k: v for k, v in (structured_summary or {}).items() if k != "per_sample"
    }
    judge_summary = judge_out.get("summary", {}) or {}
    metrics = {
        "num_predictions": len(prediction_rows),
        "num_errors": len(errors),
        "structured_eval": structured_public,
        "llm_judge_summary": judge_summary,
    }
    write_json(layout["reports"] / "metrics.json", metrics)

    scores: Dict[str, Any] = {
        "num_predictions": len(prediction_rows),
        "num_errors": len(errors),
    }
    for k, v in structured_public.items():
        if isinstance(v, (int, float)):
            scores[f"structured.{k}"] = v
    for k, v in judge_summary.items():
        if isinstance(v, (int, float)):
            scores[f"judge.{k}"] = v
    write_json(layout["reports"] / "scores.json", scores)

    summary_lines = [
        f"prediction_path: {prediction_path}",
        f"num_predictions: {len(prediction_rows)}",
        f"num_errors: {len(errors)}",
        "",
        "== structured_eval ==",
    ]
    for k, v in structured_public.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append("== llm_judge ==")
    for k, v in judge_summary.items():
        summary_lines.append(f"  {k}: {v}")
    (layout["reports"] / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

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
