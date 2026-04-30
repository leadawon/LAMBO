"""GeneratorV2 — paired with GlobalComposerV3.

Consumes the composer v3 handoff (query_spec / doc_records / structure /
completeness / filled_skeleton) and emits the user-facing answer. The
generator does NOT redo cross-doc reasoning — its job is to verify the
filled_skeleton conforms to the instruction's exact format and emit it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..backend import GeminiClient, OpenAIClient, QwenLocalClient
from ..common import extract_json_payload, read_json, write_json


class GeneratorV2:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        pdir = prompt_dir or Path(__file__).resolve().parents[1] / "prompts" / "generate_v2"
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
        doc_title_list: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        cache_path = sample_dir / "generator.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        query_spec = composed.get("query_spec", {}) or {}
        doc_records = composed.get("doc_records", []) or []
        structure = composed.get("structure", {}) or {}
        completeness = composed.get("completeness", {}) or {}
        filled_skeleton = composed.get("filled_skeleton", None)

        title_list = doc_title_list or {}
        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            doc_title_list=json.dumps(title_list, ensure_ascii=False, indent=2),
            query_spec_json=json.dumps(query_spec, ensure_ascii=False, indent=2),
            doc_records_json=json.dumps(doc_records, ensure_ascii=False, indent=2),
            structure_json=json.dumps(structure, ensure_ascii=False, indent=2),
            completeness_json=json.dumps(completeness, ensure_ascii=False, indent=2),
            filled_skeleton_json=json.dumps(
                filled_skeleton, ensure_ascii=False, indent=2
            ),
        )

        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=8192,
            metadata={"module": "generator_v2"},
        )

        final_answer: Any = raw_text.strip()
        parsed = extract_json_payload(raw_text)
        if parsed is not None:
            final_answer = parsed

        result = {
            "final_answer": final_answer,
            "raw_text": raw_text,
            "filled_skeleton": filled_skeleton,
            "ref_unit": (query_spec.get("projector") or {}).get("ref_unit", ""),
        }
        write_json(cache_path, result)
        return result
