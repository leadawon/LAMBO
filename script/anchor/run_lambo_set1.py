from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.anchor.anchor_agent import AnchorAgent
from script.anchor.answer_writer import AnswerWriter
from script.anchor.backend import get_default_client
from script.anchor.common import (
    extract_loong_score,
    json_dumps_pretty,
    read_jsonl,
    safe_filename,
    write_json,
    write_jsonl,
)
from script.anchor.evaluate_structured import evaluate_predictions
from script.anchor.manifest import build_set1_manifest, load_records, save_manifest
from script.anchor.refine_extractor import RefineExtractor
from script.anchor.search_r1 import SearchR1


DEFAULT_INPUT_PATH = PROJECT_ROOT / "Loong" / "data" / "loong_process.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "lambo_set1_10"
DEFAULT_LOONG_SRC = PROJECT_ROOT / "Loong" / "src"
DEFAULT_LOONG_MODEL_DIR = PROJECT_ROOT / "Loong" / "config" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LAMBO set1-10 anchor experiment.")
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--process_num_eval", type=int, default=5)
    parser.add_argument("--disable_llm_search", action="store_true")
    return parser.parse_args()


def ensure_output_layout(output_dir: Path) -> Dict[str, Path]:
    layout = {
        "root": output_dir,
        "samples": output_dir / "samples",
        "reports": output_dir / "reports",
        "judge": output_dir / "judge",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def run_loong_judge(
    *,
    prediction_path: Path,
    judge_output_path: Path,
    process_num_eval: int,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(DEFAULT_LOONG_SRC / "step3_model_evaluate.py"),
        "--models",
        "qwen2.yaml",
        "--eval_model",
        "gpt4o.yaml",
        "--debug_num",
        "-1",
        "--doc_path",
        str(PROJECT_ROOT / "Loong" / "data" / "doc"),
        "--input_path",
        str(PROJECT_ROOT / "Loong" / "data" / "loong.jsonl"),
        "--output_process_path",
        str(PROJECT_ROOT / "Loong" / "data" / "loong_process.jsonl"),
        "--output_path",
        str(prediction_path),
        "--evaluate_output_path",
        str(judge_output_path),
        "--max_length",
        "128000",
        "--model_config_dir",
        str(DEFAULT_LOONG_MODEL_DIR),
        "--process_num_eval",
        str(process_num_eval),
    ]
    subprocess.run(command, cwd=str(DEFAULT_LOONG_SRC), check=True)

    metric_command = [
        sys.executable,
        str(DEFAULT_LOONG_SRC / "step4_cal_metric.py"),
        "--models",
        "qwen2.yaml",
        "--eval_model",
        "gpt4o.yaml",
        "--debug_num",
        "-1",
        "--doc_path",
        str(PROJECT_ROOT / "Loong" / "data" / "doc"),
        "--input_path",
        str(PROJECT_ROOT / "Loong" / "data" / "loong.jsonl"),
        "--output_process_path",
        str(PROJECT_ROOT / "Loong" / "data" / "loong_process.jsonl"),
        "--output_path",
        str(prediction_path),
        "--evaluate_output_path",
        str(judge_output_path),
        "--max_length",
        "128000",
        "--model_config_dir",
        str(DEFAULT_LOONG_MODEL_DIR),
        "--process_num_eval",
        str(process_num_eval),
    ]
    subprocess.run(metric_command, cwd=str(DEFAULT_LOONG_SRC), check=True)

    judge_rows = read_jsonl(judge_output_path)
    scores = []
    per_sample = []
    for row in judge_rows:
        score = extract_loong_score(row.get("eval_response", ""))
        if score is not None:
            scores.append(score)
        per_sample.append(
            {
                "id": row.get("id"),
                "selected_index": row.get("selected_index"),
                "score": score,
            }
        )
    return {
        "sample_count": len(judge_rows),
        "scoring_success_rate": len(scores) / len(judge_rows) if judge_rows else 0.0,
        "avg_score": mean(scores) if scores else None,
        "perfect_rate": sum(1 for score in scores if score == 100) / len(scores) if scores else 0.0,
        "per_sample": per_sample,
    }


def build_report(
    *,
    manifest_rows: List[Dict[str, Any]],
    structured_summary: Dict[str, Any],
    judge_summary: Dict[str, Any] | None,
) -> str:
    lines = ["# LAMBO Set1-10 Report", ""]
    lines.append("## Manifest")
    for item in manifest_rows:
        lines.append(
            f"- idx={item['selected_index']} | {item['record_type']} | level={item['level']} | sample_id={item['sample_id']}"
        )
    lines.append("")
    lines.append("## Structured Metrics")
    lines.append(f"- sample_count: {structured_summary['sample_count']}")
    lines.append(f"- exact_match_rate: {structured_summary['exact_match_rate']:.4f}")
    lines.append(f"- avg_pair_f1: {structured_summary['avg_pair_f1'] if structured_summary['avg_pair_f1'] is not None else 'N/A'}")
    lines.append("")
    lines.append("## Structured Per Sample")
    for row in structured_summary["per_sample"]:
        lines.append(
            f"- idx={row['selected_index']} | exact_match={row['exact_match']} | pair_f1={row['pair_f1']} | missing={row['missing']} | extra={row['extra']}"
        )
    lines.append("")
    lines.append("## Loong Judge")
    if judge_summary is None:
        lines.append("- judge skipped or failed")
    else:
        lines.append(f"- sample_count: {judge_summary['sample_count']}")
        lines.append(f"- scoring_success_rate: {judge_summary['scoring_success_rate']:.4f}")
        lines.append(f"- avg_score: {judge_summary['avg_score']}")
        lines.append(f"- perfect_rate: {judge_summary['perfect_rate']:.4f}")
        for row in judge_summary["per_sample"]:
            lines.append(f"- idx={row['selected_index']} | score={row['score']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    layout = ensure_output_layout(output_dir)

    records = load_records(input_path)
    manifest = build_set1_manifest(records)
    if args.max_items is not None:
        manifest = manifest[: args.max_items]
    manifest_path = save_manifest(manifest, layout["root"] / "manifest.json")

    llm = get_default_client()
    anchor_agent = AnchorAgent(llm=llm)
    search_r1 = SearchR1(llm=llm, use_llm_planning=not args.disable_llm_search)
    refine_extractor = RefineExtractor(llm=llm, search_agent=search_r1)
    answer_writer = AnswerWriter(llm=llm)

    prediction_rows: List[Dict[str, Any]] = []
    for item in manifest:
        record = records[item.selected_index]
        sample_dir = layout["samples"] / safe_filename(item.sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        anchor_payload = anchor_agent.run(record=record, sample_dir=sample_dir, force=args.force)
        doc_results: List[Dict[str, Any]] = []
        for doc_payload in anchor_payload["docs"]:
            search_output = search_r1.run(
                record=record,
                doc_payload=doc_payload,
                sample_dir=sample_dir,
                force=args.force,
            )
            refine_output = refine_extractor.run(
                record=record,
                doc_payload=doc_payload,
                search_output=search_output,
                sample_dir=sample_dir,
                force=args.force,
            )
            doc_results.append(refine_output)

        answer_output = answer_writer.run(
            record=record,
            doc_results=doc_results,
            sample_dir=sample_dir,
            force=args.force,
        )

        prediction_row = dict(record)
        prediction_row["selected_index"] = item.selected_index
        prediction_row["sample_id"] = item.sample_id
        prediction_row["generate_response"] = answer_output["final_answer"]
        prediction_row["lambo_trace_dir"] = str(sample_dir)
        prediction_rows.append(prediction_row)

    prediction_path = write_jsonl(layout["root"] / "lambo_predictions.jsonl", prediction_rows)
    structured_summary = evaluate_predictions(prediction_rows)
    structured_summary_path = write_json(layout["reports"] / "structured_eval.json", structured_summary)

    judge_summary = None
    if not args.skip_judge:
        judge_output_path = layout["judge"] / "loong_evaluate.jsonl"
        try:
            judge_summary = run_loong_judge(
                prediction_path=prediction_path,
                judge_output_path=judge_output_path,
                process_num_eval=args.process_num_eval,
            )
            write_json(layout["reports"] / "judge_summary.json", judge_summary)
        except Exception as exc:
            judge_summary = {"error": str(exc)}
            write_json(layout["reports"] / "judge_error.json", judge_summary)
            judge_summary = None

    report_text = build_report(
        manifest_rows=[item.to_dict() for item in manifest],
        structured_summary=structured_summary,
        judge_summary=judge_summary,
    )
    report_path = layout["reports"] / "summary_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(
        json_dumps_pretty(
            {
                "manifest_path": str(manifest_path),
                "prediction_path": str(prediction_path),
                "structured_summary_path": str(structured_summary_path),
                "report_path": str(report_path),
            }
        )
    )


if __name__ == "__main__":
    main()
