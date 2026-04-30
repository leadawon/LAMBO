"""Run the LAMBO v3 pipeline (AnchorAgentV2 + DocRefineAgentV2 + GlobalComposerV3 + GeneratorV2).

Pipeline: AnchorAgentV2 → DocRefineAgentV2 (per-doc) → GlobalComposerV3 → GeneratorV2
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

from lambo_v2.agents.anchor_agent_v2 import AnchorAgentV2
from lambo_v2.agents.doc_refine_agent_v2 import DocRefineAgentV2
from lambo_v2.agents.global_composer_v3 import GlobalComposerV3
from lambo_v2.agents.generator_v2 import GeneratorV2
from lambo_v2.backend import get_default_client
from lambo_v2.common import (
    current_query_from_record,
    ensure_dir,
    instruction_from_record,
    json_dumps_pretty,
    safe_filename,
    write_json,
    write_jsonl,
)
from lambo_v2.eval.structured_eval import evaluate_predictions
from lambo_v2.eval.llm_judge import run_llm_judge
from lambo_v2.manifest import build_set1_manifest, load_records, save_manifest


DEFAULT_INPUT_PATH = PROJECT_ROOT / "reference" / "Loong" / "data" / "loong_process.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "lambo_v3_toc_10"

MAX_OUTPUT_TOKENS = 50000
MAX_INPUT_TOKENS = 50000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LAMBO v3 (composer v3 + generator v2) pipeline.")
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument(
        "--selected_indices",
        type=str,
        default="",
        help="Comma-separated selected_index values for targeted experiments.",
    )
    parser.add_argument(
        "--selected_indices_path",
        type=str,
        default="",
        help="Path to JSON file containing indices list (dict with 'indices' key or plain list).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_refine_rounds", type=int, default=6)
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["local", "gemini", "openai"],
        help="LLM backend: 'openai' for vLLM server (default), 'gemini', 'local'",
    )
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

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    layout = ensure_layout(output_dir)

    records = load_records(input_path)
    wanted = set()
    if args.selected_indices_path.strip():
        raw = json.load(open(args.selected_indices_path))
        idx_list = raw["indices"] if isinstance(raw, dict) and "indices" in raw else raw
        wanted = {int(i) for i in idx_list}
    if args.selected_indices.strip():
        wanted |= {
            int(part.strip())
            for part in args.selected_indices.split(",")
            if part.strip()
        }
    if wanted:
        from lambo_v2.manifest import ManifestItem
        manifest = []
        for idx in sorted(wanted):
            if idx >= len(records):
                continue
            rec = records[idx]
            manifest.append(ManifestItem(
                selected_index=idx,
                record_id=str(rec.get("id", "unknown")),
                set_id=int(rec.get("set", 0) or 0),
                record_type=str(rec.get("type", "unknown")),
                level=int(rec.get("level", 0) or 0),
                language=str(rec.get("language", "unknown")),
                question=str(rec.get("question", "")).strip(),
                sample_id=f"{rec.get('type', 'unknown')}_level{rec.get('level', 0)}_{idx}",
            ))
    else:
        manifest = build_set1_manifest(records)
    if args.max_items is not None:
        manifest = manifest[: args.max_items]
    save_manifest(manifest, layout["root"] / "manifest.json")

    print(f"Backend: {args.backend}", flush=True)
    llm = get_default_client(backend=args.backend)

    if hasattr(llm, '_default_max_output_tokens'):
        llm._default_max_output_tokens = MAX_OUTPUT_TOKENS

    anchor_agent = AnchorAgentV2(llm=llm)
    doc_refine_agent = DocRefineAgentV2(
        llm=llm,
        max_rounds=args.max_refine_rounds,
    )
    composer = GlobalComposerV3(llm=llm)
    generator = GeneratorV2(llm=llm)

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

            anchor_payload = anchor_agent.run(
                record=record, sample_dir=sample_dir, force=args.force,
            )

            doc_sheets: List[Dict[str, Any]] = []
            other_docs_list = [
                {"doc_id": d["doc_id"], "doc_title": d.get("doc_title", "")}
                for d in anchor_payload["docs"]
            ]
            for doc_payload in anchor_payload["docs"]:
                toc_count = doc_payload.get("toc_count", len(doc_payload.get("toc", [])))
                print(
                    f"  doc {doc_payload['doc_id']} '{doc_payload['doc_title'][:40]}' "
                    f"toc_sections={toc_count}",
                    flush=True,
                )
                doc_refine_agent.max_rounds = min(
                    args.max_refine_rounds,
                    max(1, toc_count + 1),
                )
                sheet = doc_refine_agent.run(
                    question=question,
                    instruction=instruction,
                    doc_payload=doc_payload,
                    sample_dir=sample_dir,
                    force=args.force,
                    other_docs=other_docs_list,
                )
                doc_sheets.append(sheet)
                evidence_preview = (sheet.get("evidence", "") or "")[:80]
                print(
                    f"    -> {sheet['scan_result']} "
                    f"| rounds={sheet.get('rounds_used', 0)} "
                    f"| evidence={evidence_preview!r}",
                    flush=True,
                )

            composed = composer.run(
                question=question,
                instruction=instruction,
                doc_sheets=doc_sheets,
                anchor_docs=anchor_payload["docs"],
                sample_dir=sample_dir,
                force=args.force,
            )
            ref_unit = (
                (composed.get("query_spec") or {}).get("projector") or {}
            ).get("ref_unit", "")
            structure_form = (composed.get("structure") or {}).get("form", "")
            warnings = composed.get("warnings") or []
            print(
                f"  composer -> ref_unit={ref_unit} | structure={structure_form} "
                f"| match_count={len([r for r in composed.get('doc_records', []) if r.get('verdict') == 'match'])} "
                f"| warnings={len(warnings)}",
                flush=True,
            )
            if warnings:
                for w in warnings:
                    print(f"    !! {w}", flush=True)

            doc_title_list = {
                d["doc_id"]: d.get("doc_title", d["doc_id"])
                for d in anchor_payload["docs"]
            }
            gen_out = generator.run(
                question=question,
                instruction=instruction,
                composed=composed,
                sample_dir=sample_dir,
                force=args.force,
                doc_title_list=doc_title_list,
            )
            final_answer = gen_out["final_answer"]

            keep = ("id", "type", "level", "question", "instruction", "answer")
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
