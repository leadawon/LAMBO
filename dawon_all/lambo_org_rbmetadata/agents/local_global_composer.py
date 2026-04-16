"""LocalGlobalComposer — accept dict-valued ``records`` without dropping them.

``lambo.agents.global_composer.GlobalComposer`` was tightened to only accept a
``list``-shaped ``records`` field. Loong SET1 legal classification samples
(e.g. ``legal_level3_4xx``) ask the LLM to bucket court documents by case type
and the LLM naturally replies with a dict like
``{"行政案件": ["《判决文书2》"], ...}``. Dropping those drops the answer.

This subclass:
  * keeps dict-shaped ``records`` as-is,
  * if ``records`` came back empty but ``raw_text`` contains a dict under
    ``records``, re-parses and recovers it,
  * otherwise behaves identically to the upstream composer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from lambo.agents.global_composer import GlobalComposer
from lambo.backend import QwenLocalClient, GeminiClient, OpenAIClient
from lambo.common import extract_json_payload, read_json, write_json


class LocalGlobalComposer(GlobalComposer):
    def __init__(
        self,
        llm: Union[QwenLocalClient, GeminiClient, OpenAIClient],
        prompt_dir: Optional[Path] = None,
    ) -> None:
        super().__init__(llm=llm, prompt_dir=prompt_dir)

    def run(self, *args, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        result = super().run(*args, **kwargs)
        records = result.get("records")
        raw_text = result.get("raw_text", "")
        # Already non-empty list or already a dict — nothing to fix.
        if isinstance(records, list) and records:
            return result
        if isinstance(records, dict) and records:
            return result
        # Attempt salvage from raw_text.
        payload = extract_json_payload(raw_text)
        if isinstance(payload, dict):
            raw_records = payload.get("records")
            if isinstance(raw_records, dict) and raw_records:
                result["records"] = raw_records
            elif isinstance(raw_records, list) and raw_records:
                result["records"] = raw_records
        # Persist salvage.
        sample_dir = kwargs.get("sample_dir")
        if sample_dir is not None:
            write_json(Path(sample_dir) / "composed.json", result)
        return result
