from __future__ import annotations

import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from .common import extract_json_payload

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# ---------------------------------------------------------------------------
# Local Qwen backend (Transformers)
# ---------------------------------------------------------------------------
META_COGNITIVE_SRC = Path("/workspace/meta-cognitive-RAG/src")
if str(META_COGNITIVE_SRC) not in sys.path:
    sys.path.insert(0, str(META_COGNITIVE_SRC))


class QwenLocalClient:
    def __init__(
        self,
        model_dir: Path | str = "/workspace/meta-cognitive-RAG/models/Qwen2.5-32B-Instruct",
        max_output_tokens: int = 2048,
        max_input_tokens: int = 120000,
        per_gpu_max_memory_gib: int = 22,
    ) -> None:
        from meta_cognitive_rag.local_backend import LocalTransformersBackend, LocalTransformersConfig  # type: ignore
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


# ---------------------------------------------------------------------------
# Gemini API backend
# ---------------------------------------------------------------------------
class GeminiClient:
    """Google Gemini API client with the same interface as QwenLocalClient."""

    MAX_RETRIES = 5
    INITIAL_BACKOFF = 2.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        import google.generativeai as genai

        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Set it in .env or pass api_key= to GeminiClient()."
            )
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17")
        genai.configure(api_key=self.api_key)
        self._genai = genai

    def _call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt if system_prompt else None,
        )
        gen_config = self._genai.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        backoff = self.INITIAL_BACKOFF
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = model.generate_content(
                    user_prompt,
                    generation_config=gen_config,
                )
                return response.text
            except Exception as exc:
                err_str = str(exc).lower()
                retryable = any(
                    kw in err_str
                    for kw in ("429", "resource_exhausted", "quota", "rate", "500", "503", "unavailable", "deadline")
                )
                if retryable and attempt < self.MAX_RETRIES:
                    print(
                        f"  [Gemini] retry {attempt}/{self.MAX_RETRIES} after {backoff:.0f}s: {exc}",
                        flush=True,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return self._call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens or 2048,
            temperature=temperature,
        )

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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_client_cache: dict[str, Any] = {}


def get_default_client(backend: str = "local") -> QwenLocalClient | GeminiClient:
    """Return a cached LLM client.

    Args:
        backend: "local" for QwenLocalClient, "gemini" for GeminiClient.
    """
    if backend in _client_cache:
        return _client_cache[backend]

    if backend == "gemini":
        client = GeminiClient()
    else:
        client = QwenLocalClient()

    _client_cache[backend] = client
    return client
