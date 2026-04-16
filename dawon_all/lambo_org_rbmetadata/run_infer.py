"""Run the lambo_org retrieval baseline on the Loong SET1 99-example subset.

Mirrors dawonv8/run_set1.py and dawonv7/anchor/run_lambo_set1.py, but:
- replaces the LLM-summary AnchorAgent with :class:`RetrievalAnchorAgent`
- replaces the iterative DocRefineAgent with :class:`RetrievalDocRefineAgent`
- keeps GlobalComposer and Generator from the original ``lambo`` package so
  the end-to-end prediction schema matches existing evaluation scripts.

The original ``lambo`` package is NOT modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


# Make both the `lambo` package and the `dawon_all` namespace importable.
PKG_DIR = Path(__file__).resolve().parent
DAWON_ROOT = PKG_DIR.parent
LAMBO_ROOT = DAWON_ROOT.parent
for candidate in (LAMBO_ROOT, DAWON_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

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

from lambo_org_rbmetadata.agents import (
    LocalGlobalComposer,
    RetrievalAnchorAgent,
    RetrievalDocRefineAgent,
)
from lambo_org_rbmetadata.backend import get_default_client
from lambo_org_rbmetadata.config import (
    DEFAULT_INDICES_OUTPUT,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    LamboOrgConfig,
)
from lambo_org_rbmetadata.embeddings import EmbeddingBackend
from lambo_org_rbmetadata.manifest import (
    build_manifest_for_indices,
    build_set1_manifest,
    load_records,
    load_selected_indices,
    save_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lambo_org retrieval baseline on Loong SET1 samples."
    )
    parser.add_argument("--input_path", type=str, default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--selected_indices", type=str, default="")
    parser.add_argument(
        "--selected_indices_path",
        type=str,
        default=str(DEFAULT_INDICES_OUTPUT),
        help="Path to indices JSON. Defaults to the 99-example subset file.",
    )
    parser.add_argument("--embedding_model", type=str, default=LamboOrgConfig.embedding_model)
    parser.add_argument("--top_k_per_doc", type=int, default=LamboOrgConfig.top_k_per_doc)
    parser.add_argument("--top_k_global", type=int, default=LamboOrgConfig.top_k_global)
    parser.add_argument(
        "--retrieval_scope",
        type=str,
        default=LamboOrgConfig.retrieval_scope,
        choices=["per_document", "global"],
    )
    parser.add_argument(
        "--backend", type=str, default="local", choices=["local", "gemini"]
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
    output_dir = Path(args.output_dir)
    layout = ensure_layout(output_dir)

    records = load_records(input_path)
    selected_indices = load_selected_indices(args.selected_indices, args.selected_indices_path)
    if selected_indices is None:
        manifest = build_set1_manifest(records)
    else:
        manifest = build_manifest_for_indices(records, selected_indices)
    if args.max_items is not None:
        manifest = manifest[: args.max_items]
    save_manifest(manifest, layout["root"] / "manifest.json")

    print(f"lambo_org | backend={args.backend} | manifest_size={len(manifest)}", flush=True)
    print(
        f"  embedding_model={args.embedding_model} scope={args.retrieval_scope} "
        f"top_k_per_doc={args.top_k_per_doc} top_k_global={args.top_k_global}",
        flush=True,
    )

    llm = get_default_client(backend=args.backend)
    embedder = EmbeddingBackend(model_name=args.embedding_model)

    anchor_agent = RetrievalAnchorAgent(
        embedder=embedder,
        top_k_per_doc=args.top_k_per_doc,
        top_k_global=args.top_k_global,
        retrieval_scope=args.retrieval_scope,
    )
    doc_refine_agent = RetrievalDocRefineAgent(top_k_per_doc=args.top_k_per_doc)
    composer = LocalGlobalComposer(llm=llm)
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
            instruction = instruction_from_record(str(record.get("instruction", "")).strip())

            anchor_payload = anchor_agent.run(
                record=record, sample_dir=sample_dir, force=args.force
            )

            doc_sheets: List[Dict[str, Any]] = []
            for doc_payload in anchor_payload["docs"]:
                print(
                    f"  doc {doc_payload['doc_id']} '{doc_payload['doc_title'][:40]}' "
                    f"chunks={doc_payload['anchor_count']}",
                    flush=True,
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
                print(
                    f"    -> {sheet['scan_result']} | "
                    f"retrieved={len(sheet.get('retrieved_anchors', []))}",
                    flush=True,
                )

            composed = composer.run(
                question=question,
                instruction=instruction,
                doc_sheets=doc_sheets,
                sample_dir=sample_dir,
                force=args.force,
            )

            gen_out = generator.run(
                question=question,
                instruction=instruction,
                composed=composed,
                sample_dir=sample_dir,
                force=args.force,
            )
            final_answer = gen_out["final_answer"]
            if (
                isinstance(final_answer, str)
                and final_answer.startswith("{{")
                and final_answer.endswith("}}")
            ):
                inner = "{" + final_answer[2:-2] + "}"
                try:
                    parsed = json.loads(inner)
                    final_answer = parsed
                    gen_out["final_answer"] = parsed
                    gen_out["raw_text"] = inner
                    gen_out["salvaged_from_double_brace"] = True
                    write_json(sample_dir / "generator.json", gen_out)
                except Exception:
                    pass

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
    write_json(layout["reports"] / "errors.json", {"count": len(errors), "errors": errors})
    print(
        json_dumps_pretty(
            {
                "prediction_path": str(prediction_path),
                "num_predictions": len(prediction_rows),
                "num_errors": len(errors),
                "output_dir": str(layout["root"]),
            }
        )
    )


if __name__ == "__main__":
    main()
