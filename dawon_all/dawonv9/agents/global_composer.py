"""dawonv8 GlobalComposer — prepend deterministic cross-doc citation block.

v7 observed that the LLM composer rarely surfaces cross-document citations
on its own. We run a deterministic title-match pass over per-doc evidence
BEFORE calling the composer, then inject the findings into the prompt so
the LLM can build on real signal instead of rediscovering it.
"""

from __future__ import annotations

from typing import Any, Dict, List

from lambo.agents.global_composer import GlobalComposer

from ..enrichment.cross_doc_matcher import (
    detect_cross_doc_citations,
    format_for_refiner,
)


def _sheet_to_matcher_doc_result(sheet: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a DocRefineAgent sheet into the shape the matcher expects."""
    evidence = str(sheet.get("evidence", "")).strip()
    items = [{"fact": evidence}] if evidence else []
    return {"doc_id": sheet.get("doc_id", ""), "items": items}


class CrossDocGlobalComposer(GlobalComposer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_cross_doc: Dict[str, Any] = {}

    def _prepare_sheets_for_prompt(self, doc_sheets: List[Dict[str, Any]]) -> str:
        base = super()._prepare_sheets_for_prompt(doc_sheets)

        docs = [
            {"doc_id": s.get("doc_id", ""), "doc_title": s.get("doc_title", "")}
            for s in doc_sheets
        ]
        doc_results = [_sheet_to_matcher_doc_result(s) for s in doc_sheets]
        cross_doc = detect_cross_doc_citations(docs=docs, doc_results=doc_results)
        self._last_cross_doc = cross_doc

        cross_block = format_for_refiner(cross_doc)
        return f"{cross_block}\n\n{base}"

    def run(
        self,
        *,
        question: str,
        instruction: str,
        doc_sheets: List[Dict[str, Any]],
        sample_dir,
        force: bool = False,
    ) -> Dict[str, Any]:
        result = super().run(
            question=question,
            instruction=instruction,
            doc_sheets=doc_sheets,
            sample_dir=sample_dir,
            force=force,
        )
        # Persist cross-doc analysis alongside the composed output for
        # offline inspection / debugging.
        if self._last_cross_doc:
            result.setdefault("cross_doc_citations", self._last_cross_doc)
        return result
