"""Generator — final answer serialization.

Consumes the Composer's tagged plain-text package
    reading_trace / evidence_units / evidence_relations / answer_basis /
    uncertainty
and emits the user-facing answer in the exact format requested by the
instruction. The Composer never prescribes the answer's surface form;
the Generator reads the instruction and picks the right surface from
the answer_basis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..backend import QwenLocalClient, GeminiClient, OpenAIClient
from ..common import extract_json_payload, read_json, write_json


class Generator:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        pdir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts" / "generate"
        self.system_prompt = (pdir / "system.txt").read_text(encoding="utf-8").strip()
        self.user_template = (pdir / "user.txt").read_text(encoding="utf-8").strip()

    def run(
        self,
        *,
        question: str,
        instruction: str,
        composed: Dict[str, Any],
        sample_dir: Path,
        force: bool = False,
        doc_title_list: Optional[Dict[str, str]] = None,  # accepted for backward compat; ignored
    ) -> Dict[str, Any]:
        cache_path = sample_dir / "generator.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        # All doc identity (doc_id + doc_title) and entity material is
        # carried inside the Composer's evidence_units / answer_basis.
        # We do NOT pass a separate doc_title bundle — the Generator
        # reads identities from the Composer output itself.
        reading_trace      = composed.get("reading_trace", "")
        evidence_units     = composed.get("evidence_units", "")
        evidence_relations = composed.get("evidence_relations", "")
        answer_basis       = composed.get("answer_basis", "")
        uncertainty        = composed.get("uncertainty", "")

        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            reading_trace=reading_trace,
            evidence_units=evidence_units,
            evidence_relations=evidence_relations,
            answer_basis=answer_basis,
            uncertainty=uncertainty,
        )

        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=16384,
            metadata={"module": "generator"},
        )

        # Try to parse as JSON first; fall back to the raw string for unstructured answers
        final_answer: Any = raw_text.strip()
        parsed = extract_json_payload(raw_text)
        if parsed is not None:
            final_answer = parsed

        result = {
            "final_answer": final_answer,
            "raw_text": raw_text,
            "answer_basis": answer_basis,
        }
        write_json(cache_path, result)
        return result
