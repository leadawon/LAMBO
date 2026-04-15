"""Evaluate predictions produced by ``run_infer.py``.

Uses the same evaluation path as dawonv8 (``lambo.eval.structured_eval`` and
``lambo.eval.llm_judge``) so results are directly comparable. Final scores
are written in three forms:

- ``reports/metrics.json`` — machine-readable structured eval + judge summary
- ``reports/scores.json`` — flat {metric: value} view for quick diffing
- ``reports/summary.txt`` — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PKG_DIR = Path(__file__).resolve().parent
DAWON_ROOT = PKG_DIR.parent
LAMBO_ROOT = DAWON_ROOT.parent
for candidate in (LAMBO_ROOT, DAWON_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lambo.common import ensure_dir, read_jsonl, write_json
from lambo_org.backend import get_default_client
from lambo.eval.llm_judge import run_llm_judge
from lambo.eval.structured_eval import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate lambo_org predictions.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Optional explicit predictions path. Defaults to "
        "<output_dir>/lambo_predictions.jsonl.",
    )
    parser.add_argument(
        "--skip_judge",
        action="store_true",
        help="Skip the LLM-judge pass (structured eval only).",
    )
    parser.add_argument("--backend", type=str, default="local", choices=["local", "gemini"])
    return parser.parse_args()


def _decode_row(row: Dict[str, Any]) -> Dict[str, Any]:
    response = row.get("generate_response")
    if isinstance(response, str) and response.startswith(("{", "[")):
        try:
            response = json.loads(response)
        except Exception:
            pass
    return {**row, "generate_response": response}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reports_dir = output_dir / "reports"
    ensure_dir(reports_dir)

    predictions_path = Path(args.predictions) if args.predictions else output_dir / "lambo_predictions.jsonl"
    if not predictions_path.exists():
        raise SystemExit(f"No predictions at {predictions_path}")

    rows: List[Dict[str, Any]] = read_jsonl(predictions_path)
    decoded = [_decode_row(row) for row in rows]

    try:
        structured_summary = evaluate_predictions(decoded)
    except Exception as exc:  # noqa: BLE001
        structured_summary = {"error": str(exc)}
    write_json(reports_dir / "structured_eval.json", structured_summary)

    judge_out: Dict[str, Any]
    if args.skip_judge:
        judge_out = {"summary": {"skipped": True}, "verdicts": []}
    else:
        try:
            llm = get_default_client(backend=args.backend)
            judge_out = run_llm_judge(llm=llm, prediction_rows=decoded)
        except Exception as exc:  # noqa: BLE001
            judge_out = {"summary": {"error": str(exc)}, "verdicts": []}
    write_json(reports_dir / "llm_judge.json", judge_out)

    structured_public = {k: v for k, v in (structured_summary or {}).items() if k != "per_sample"}
    metrics = {
        "num_predictions": len(rows),
        "structured_eval": structured_public,
        "llm_judge_summary": judge_out.get("summary", {}),
    }
    write_json(reports_dir / "metrics.json", metrics)

    scores: Dict[str, Any] = {"num_predictions": len(rows)}
    for key, value in structured_public.items():
        if isinstance(value, (int, float)):
            scores[f"structured.{key}"] = value
    for key, value in (judge_out.get("summary") or {}).items():
        if isinstance(value, (int, float)):
            scores[f"judge.{key}"] = value
    write_json(reports_dir / "scores.json", scores)

    summary_lines = [
        f"predictions: {predictions_path}",
        f"num_predictions: {len(rows)}",
        "",
        "== structured_eval ==",
    ]
    for key, value in structured_public.items():
        summary_lines.append(f"  {key}: {value}")
    summary_lines.append("")
    summary_lines.append("== llm_judge ==")
    for key, value in (judge_out.get("summary") or {}).items():
        summary_lines.append(f"  {key}: {value}")
    summary_path = reports_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"metrics: {reports_dir / 'metrics.json'}")
    print(f"scores:  {reports_dir / 'scores.json'}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
