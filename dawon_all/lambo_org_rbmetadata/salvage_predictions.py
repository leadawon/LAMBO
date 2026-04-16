"""Offline salvage for composed.records that came back as a dict.

The upstream ``GlobalComposer`` was tightened to only accept list-shaped
``records``. Loong SET1 classification samples (e.g. ``legal_level3_4xx``)
naturally produce dict-shaped records like ``{"行政案件": [...], ...}``.

This script walks an existing run directory, re-parses ``raw_text`` for each
sample whose ``records`` came back empty, recovers dict-shaped records, and
rewrites:
  * ``composed.json``
  * ``generator.json``      (final_answer := JSON(dict records) for these)
  * ``lambo_predictions.jsonl``

No LLM calls — pure re-parsing of saved raw_text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make `lambo` importable
PKG_DIR = Path(__file__).resolve().parent
LAMBO_ROOT = PKG_DIR.parents[1]
if str(LAMBO_ROOT) not in sys.path:
    sys.path.insert(0, str(LAMBO_ROOT))

from lambo.common import extract_json_payload, read_json, write_json


def _salvage_composed(sample_dir: Path) -> bool:
    """Return True if composed.json was rewritten with a dict-records."""
    cp = sample_dir / "composed.json"
    if not cp.exists():
        return False
    composed = read_json(cp)
    records = composed.get("records")
    if isinstance(records, dict) and records:
        return False
    if isinstance(records, list) and records:
        return False
    raw_text = composed.get("raw_text", "")
    payload = extract_json_payload(raw_text)
    if not isinstance(payload, dict):
        return False
    raw_records = payload.get("records")
    if not isinstance(raw_records, dict) or not raw_records:
        return False
    composed["records"] = raw_records
    if isinstance(payload.get("structure_description"), str) and not composed.get("structure_description"):
        composed["structure_description"] = payload["structure_description"]
    write_json(cp, composed)
    return True


def _salvage_double_brace(sample_dir: Path) -> bool:
    """Fix ``generator.json`` when the LLM emitted ``{{...}}`` (double-braced
    JSON). Strips one brace layer and re-parses; on success, overwrites
    ``final_answer`` and ``raw_text`` with the valid structure.
    """
    gp = sample_dir / "generator.json"
    if not gp.exists():
        return False
    gen = read_json(gp)
    fa = gen.get("final_answer")
    if not (isinstance(fa, str) and fa.startswith("{{") and fa.endswith("}}")):
        return False
    inner = "{" + fa[2:-2] + "}"
    try:
        parsed = json.loads(inner)
    except Exception:
        return False
    gen["final_answer"] = parsed
    gen["raw_text"] = inner
    gen["salvaged_from_double_brace"] = True
    write_json(gp, gen)
    return True


def _salvage_generator(sample_dir: Path) -> bool:
    """If composed has dict-records but generator.final_answer is an empty
    shell (or list), overwrite final_answer with the dict records (the
    generator's job on classification tasks is just to echo the dict).
    """
    cp = sample_dir / "composed.json"
    gp = sample_dir / "generator.json"
    if not (cp.exists() and gp.exists()):
        return False
    composed = read_json(cp)
    records = composed.get("records")
    if not isinstance(records, dict) or not records:
        return False
    gen = read_json(gp)
    final_answer = gen.get("final_answer")
    # An empty shell is a dict with the same keys but all-empty list values.
    is_empty_shell = (
        isinstance(final_answer, dict)
        and set(final_answer.keys()) == set(records.keys())
        and all(
            (isinstance(v, list) and not v) or v is None
            for v in final_answer.values()
        )
    )
    # Or the LLM returned something unrelated (e.g. a list)
    needs_overwrite = is_empty_shell or (
        not isinstance(final_answer, dict) and len(str(final_answer)) < 100
    )
    if not needs_overwrite:
        return False
    gen["final_answer"] = records
    gen["salvaged_from_dict_records"] = True
    write_json(gp, gen)
    return True


def _rewrite_predictions(output_dir: Path) -> int:
    """Rewrite ``lambo_predictions.jsonl`` using on-disk generator.json values."""
    pred_path = output_dir / "lambo_predictions.jsonl"
    if not pred_path.exists():
        return 0
    rows: List[Dict[str, Any]] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    updated = 0
    safe_to_dir = {r.get("sample_id"): r.get("lambo_trace_dir", "") for r in rows}
    for row in rows:
        trace_dir = row.get("lambo_trace_dir", "")
        if not trace_dir:
            continue
        gen_path = Path(trace_dir) / "generator.json"
        if not gen_path.exists():
            continue
        gen = read_json(gen_path)
        if not (gen.get("salvaged_from_dict_records") or gen.get("salvaged_from_double_brace")):
            continue
        fa = gen.get("final_answer")
        if isinstance(fa, (dict, list)):
            row["generate_response"] = json.dumps(fa, ensure_ascii=False)
        else:
            row["generate_response"] = str(fa)
        updated += 1

    if updated:
        with pred_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    composed_fixed = 0
    generator_fixed = 0
    double_brace_fixed = 0
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        if _salvage_composed(sample_dir):
            composed_fixed += 1
        if _salvage_generator(sample_dir):
            generator_fixed += 1
        if _salvage_double_brace(sample_dir):
            double_brace_fixed += 1
    pred_updated = _rewrite_predictions(output_dir)

    print(
        json.dumps(
            {
                "composed_fixed": composed_fixed,
                "generator_fixed": generator_fixed,
                "double_brace_fixed": double_brace_fixed,
                "predictions_rewritten": pred_updated,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
