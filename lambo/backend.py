from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from .common import extract_json_payload


META_COGNITIVE_SRC = Path("/workspace/meta-cognitive-RAG/src")
if str(META_COGNITIVE_SRC) not in sys.path:
    sys.path.insert(0, str(META_COGNITIVE_SRC))

from meta_cognitive_rag.local_backend import LocalTransformersBackend, LocalTransformersConfig  # type: ignore  # noqa: E402


class QwenLocalClient:
    def __init__(
        self,
        model_dir: Path | str = "/workspace/meta-cognitive-RAG/models/Qwen2.5-32B-Instruct",
        max_output_tokens: int = 2048,
        max_input_tokens: int = 120000,
        per_gpu_max_memory_gib: int = 22,
    ) -> None:
        self.config = LocalTransformersConfig(
            model_id="Qwen/Qwen2.5-32B-Instruct",
            model_dir=model_dir,
            max_output_tokens=max_output_tokens,
            max_input_tokens=max_input_tokens,
            compute_dtype="float16",
            load_in_4bit=False,
            per_gpu_max_memory_gib=per_gpu_max_memory_gib,
        )
        self.backend = LocalTransformersBackend(self.config)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens or self.config.max_output_tokens,
            metadata=metadata or {},
        ).text

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, str]:
        raw_text = self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            metadata=metadata,
        )
        payload = extract_json_payload(raw_text)
        if payload is None:
            repair_prompt = (
                user_prompt
                + "\n\nYour previous answer was not valid JSON. Re-answer with strict JSON only. "
                + "Do not include markdown fences or commentary."
            )
            raw_text = self.generate_text(
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                metadata=metadata,
            )
            payload = extract_json_payload(raw_text)
        return payload, raw_text


@lru_cache(maxsize=1)
def get_default_client() -> QwenLocalClient:
    return QwenLocalClient()
