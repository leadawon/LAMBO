"""Generator — final answer serialization with DOC id → entity name projection.

Takes composed cross-document analysis from GlobalComposer and produces
the final human-readable answer, replacing all internal DOC* identifiers
with real entity names via the projection_map.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..backend import QwenLocalClient, GeminiClient
from ..common import (
    extract_json_payload,
    read_json,
    write_json,
)


class Generator:
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient],
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
    ) -> Dict[str, Any]:
        cache_path = sample_dir / "generator.json"
        if cache_path.exists() and not force:
            return read_json(cache_path)

        projection_map = composed.get("projection_map", {})
        records = composed.get("records", [])
        structure_description = composed.get("structure_description", "")

        composed_compact = {
            "structure_description": structure_description,
            "records": records,
        }

        user_prompt = self.user_template.format(
            question=question,
            instruction=instruction,
            projection_map=json.dumps(projection_map, ensure_ascii=False, indent=2),
            composed_json=json.dumps(composed_compact, ensure_ascii=False, indent=2),
        )

        raw_text = self.llm.generate_text(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=2000,
            metadata={"module": "generator"},
        )

        # Try to parse as JSON first
        final_answer: Any = raw_text.strip()
        parsed = extract_json_payload(raw_text)
        if parsed is not None:
            final_answer = parsed

        result = {
            "final_answer": final_answer,
            "raw_text": raw_text,
            "projection_map": projection_map,
        }
        write_json(cache_path, result)
        return result
