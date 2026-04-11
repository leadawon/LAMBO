from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

DAWON_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DAWON_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dawon.anchor.anchor_agent import AnchorAgent
from dawon.anchor.answer_writer import AnswerWriter
from dawon.anchor.backend import get_default_client
from dawon.anchor.common import (
    extract_loong_score,
    json_dumps_pretty,
    read_jsonl,
    safe_filename,
    write_json,
    write_jsonl,
)
from dawon.anchor.evaluate_structured import evaluate_predictions
from dawon.anchor.manifest import build_manifest_for_indices, build_set1_manifest, load_records, save_manifest
from dawon.anchor.paths import (
    resolve_loong_jsonl_path,
    resolve_loong_model_dir,
    resolve_loong_process_path,
    resolve_loong_src,
)
from dawon.anchor.refine_extractor import RefineExtractor
from dawon.anchor.run_loong_judge_local import run_local_judge
from dawon.anchor.search_r1 import SearchR1


DEFAULT_INPUT_PATH = resolve_loong_process_path(strict=False)
DEFAULT_OUTPUT_DIR = DAWON_ROOT / "logs" / "lambo_set1_10"
DEFAULT_LOONG_SRC = resolve_loong_src(strict=False)
DEFAULT_LOONG_MODEL_DIR = resolve_loong_model_dir(strict=False)
DEFAULT_LOONG_JSONL = resolve_loong_jsonl_path(strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LAMBO set1-10 anchor experiment.")
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH or ""))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--process_num_eval", type=int, default=5)
    parser.add_argument("--disable_llm_search", action="store_true")
    parser.add_argument("--selected_indices", type=str, default="")
    parser.add_argument("--selected_indices_path", type=str, default="")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--skip_local_judge", action="store_true")
    parser.add_argument("--local_judge_max_output_tokens", type=int, default=256)
    parser.add_argument("--judge_eval_model", type=str, default="gpt4o.yaml")
    parser.add_argument("--judge_gen_model", type=str, default="qwen2.yaml")
    parser.add_argument("--loong_src", type=str, default=str(DEFAULT_LOONG_SRC or ""))
    parser.add_argument("--loong_model_dir", type=str, default=str(DEFAULT_LOONG_MODEL_DIR or ""))
    parser.add_argument("--loong_jsonl", type=str, default=str(DEFAULT_LOONG_JSONL or ""))
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


def classify_error(exc: Exception, tb: str) -> str:
    text = f"{type(exc).__name__} {exc} {tb}".lower()
    oom_markers = [
        "out of memory",
        "cuda oom",
        "cudnn_status_alloc_failed",
        "torch.cuda.oom",
        "memoryerror",
    ]
    if any(marker in text for marker in oom_markers):
        return "oom"
    return "error"


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
    loong_src: Path,
    loong_model_dir: Path,
    loong_jsonl: Path,
    loong_process_path: Path,
    prediction_path: Path,
    judge_output_path: Path,
    process_num_eval: int,
    judge_eval_model: str,
    judge_gen_model: str,
) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(loong_src / "step3_model_evaluate.py"),
        "--models",
        judge_gen_model,
        "--eval_model",
        judge_eval_model,
        "--debug_num",
        "-1",
        "--doc_path",
        str(PROJECT_ROOT / "Loong" / "data" / "doc"),
        "--input_path",
        str(loong_jsonl),
        "--output_process_path",
        str(loong_process_path),
        "--output_path",
        str(prediction_path),
        "--evaluate_output_path",
        str(judge_output_path),
        "--max_length",
        "128000",
        "--model_config_dir",
        str(loong_model_dir),
        "--process_num_eval",
        str(process_num_eval),
    ]
    subprocess.run(command, cwd=str(loong_src), check=True)

    metric_command = [
        sys.executable,
        str(loong_src / "step4_cal_metric.py"),
        "--models",
        judge_gen_model,
        "--eval_model",
        judge_eval_model,
        "--debug_num",
        "-1",
        "--doc_path",
        str(PROJECT_ROOT / "Loong" / "data" / "doc"),
        "--input_path",
        str(loong_jsonl),
        "--output_process_path",
        str(loong_process_path),
        "--output_path",
        str(prediction_path),
        "--evaluate_output_path",
        str(judge_output_path),
        "--max_length",
        "128000",
        "--model_config_dir",
        str(loong_model_dir),
        "--process_num_eval",
        str(process_num_eval),
    ]
    subprocess.run(metric_command, cwd=str(loong_src), check=True)

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
    local_judge_summary: Dict[str, Any] | None,
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
    lines.append("## Official Judge")
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
    lines.append("## Local Judge")
    if local_judge_summary is None:
        lines.append("- local judge skipped or failed")
    else:
        lines.append(f"- sample_count: {local_judge_summary['sample_count']}")
        lines.append(f"- scoring_success_rate: {local_judge_summary['scoring_success_rate']:.4f}")
        lines.append(f"- avg_score: {local_judge_summary['avg_score']}")
        lines.append(f"- perfect_rate: {local_judge_summary['perfect_rate']:.4f}")
        for row in local_judge_summary["per_sample_scores"]:
            lines.append(f"- idx={row['selected_index']} | score={row['score']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path) if args.input_path else resolve_loong_process_path(strict=True)
    output_dir = Path(args.output_dir)
    loong_src = Path(args.loong_src) if args.loong_src else resolve_loong_src(strict=True)
    loong_model_dir = (
        Path(args.loong_model_dir) if args.loong_model_dir else resolve_loong_model_dir(strict=True)
    )
    loong_jsonl = Path(args.loong_jsonl) if args.loong_jsonl else resolve_loong_jsonl_path(strict=True)
    layout = ensure_output_layout(output_dir)

    records = load_records(input_path)
    selected_indices = load_selected_indices(args)
    if selected_indices is None:
        manifest = build_set1_manifest(records)
    else:
        manifest = build_manifest_for_indices(records, selected_indices)
    if args.max_items is not None:
        manifest = manifest[: args.max_items]
    manifest_path = save_manifest(manifest, layout["root"] / "manifest.json")

    llm = get_default_client()
    anchor_agent = AnchorAgent(llm=llm)
    search_r1 = SearchR1(llm=llm, use_llm_planning=not args.disable_llm_search)
    refine_extractor = RefineExtractor(llm=llm, search_agent=search_r1)
    answer_writer = AnswerWriter(llm=llm)

    prediction_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    error_path = layout["root"] / "sample_errors.jsonl"
    if args.force and error_path.exists():
        error_path.unlink()
    for item in manifest:
        record = records[item.selected_index]
        sample_dir = layout["samples"] / safe_filename(item.sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
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
        except Exception as exc:
            tb = traceback.format_exc()
            error_kind = classify_error(exc, tb)
            error_row = {
                "id": record.get("id"),
                "selected_index": item.selected_index,
                "sample_id": item.sample_id,
                "record_type": item.record_type,
                "level": item.level,
                "error_kind": error_kind,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": tb,
                "lambo_trace_dir": str(sample_dir),
            }
            error_rows.append(error_row)
            write_json(sample_dir / "error.json", error_row)
            write_jsonl(error_path, error_rows)

            prediction_row = dict(record)
            prediction_row["selected_index"] = item.selected_index
            prediction_row["sample_id"] = item.sample_id
            prediction_row["generate_response"] = "meet error"
            prediction_row["lambo_trace_dir"] = str(sample_dir)
            prediction_row["used_time"] = -100
            prediction_row["error_kind"] = error_kind
            prediction_row["error_type"] = type(exc).__name__
            prediction_row["error_message"] = str(exc)
            prediction_rows.append(prediction_row)
            print(
                f"Sample failed idx={item.selected_index} sample_id={item.sample_id}: {exc}",
                file=sys.stderr,
            )
            if not args.continue_on_error:
                raise

    prediction_path = write_jsonl(layout["root"] / "lambo_predictions.jsonl", prediction_rows)
    if error_rows:
        write_jsonl(error_path, error_rows)
    structured_summary = evaluate_predictions(prediction_rows)
    structured_summary_path = write_json(layout["reports"] / "structured_eval.json", structured_summary)

    judge_summary = None
    if not args.skip_judge:
        judge_output_path = layout["judge"] / "loong_evaluate.jsonl"
        if args.force and judge_output_path.exists():
            judge_output_path.unlink()
        try:
            judge_summary = run_loong_judge(
                loong_src=loong_src,
                loong_model_dir=loong_model_dir,
                loong_jsonl=loong_jsonl,
                loong_process_path=input_path,
                prediction_path=prediction_path,
                judge_output_path=judge_output_path,
                process_num_eval=args.process_num_eval,
                judge_eval_model=args.judge_eval_model,
                judge_gen_model=args.judge_gen_model,
            )
            write_json(layout["reports"] / "judge_summary.json", judge_summary)
        except Exception as exc:
            judge_summary = {"error": str(exc)}
            write_json(layout["reports"] / "judge_error.json", judge_summary)
            judge_summary = None

    local_judge_summary = None
    if not args.skip_local_judge:
        local_judge_eval_path = layout["judge"] / "loong_judge_eval_local_qwen32b.jsonl"
        local_judge_summary_path = layout["reports"] / "loong_judge_local_qwen32b.json"
        try:
            local_judge_summary = run_local_judge(
                predictions=prediction_path,
                evaluate_output=local_judge_eval_path,
                summary_output=local_judge_summary_path,
                max_output_tokens=args.local_judge_max_output_tokens,
                temperature=0.0,
            )
        except Exception as exc:
            local_judge_summary = {"error": str(exc)}
            write_json(layout["reports"] / "local_judge_error.json", local_judge_summary)
            local_judge_summary = None

    report_text = build_report(
        manifest_rows=[item.to_dict() for item in manifest],
        structured_summary=structured_summary,
        judge_summary=judge_summary,
        local_judge_summary=local_judge_summary,
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
