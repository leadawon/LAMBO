from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from .common import extract_json_payload


META_COGNITIVE_SRC = Path("/workspace/meta-cognitive-RAG/src")


def _env_first(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _load_local_backend() -> tuple[Any, Any]:
    if str(META_COGNITIVE_SRC) not in sys.path:
        sys.path.insert(0, str(META_COGNITIVE_SRC))

    from meta_cognitive_rag.local_backend import (  # type: ignore
        LocalTransformersBackend,
        LocalTransformersConfig,
    )

    return LocalTransformersBackend, LocalTransformersConfig


class _OpenAICompatChatBackend:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.chat_url = self._resolve_chat_url(self.base_url)

    @staticmethod
    def _resolve_chat_url(base_url: str) -> str:
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        return f"{base_url}/v1/chat/completions"

    @staticmethod
    def _context_retry_max_tokens(body: str, requested_tokens: int) -> Optional[int]:
        match = re.search(
            r"maximum context length is (\d+) tokens.*?"
            r"requested (\d+) output tokens.*?"
            r"prompt contains at least (\d+) input tokens",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None

        model_limit = int(match.group(1))
        requested_from_error = int(match.group(2))
        prompt_tokens = int(match.group(3))
        requested = min(requested_tokens, requested_from_error)
        margin = int(
            _env_first(
                ("LAMBO_DAWONV4_CONTEXT_RETRY_MARGIN", "LAMBO_DAWON_CONTEXT_RETRY_MARGIN"),
                "64",
            )
        )
        available = model_limit - prompt_tokens - max(0, margin)
        if available < 1:
            available = model_limit - prompt_tokens
        if 0 < available < requested:
            return available
        return None

    def _post_chat_completion(self, payload: dict[str, Any]) -> tuple[int, str]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.chat_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_output_tokens,
        }
        status, body = self._post_chat_completion(payload)
        if status == 400:
            retry_max_tokens = self._context_retry_max_tokens(body, max_output_tokens)
            if retry_max_tokens is not None:
                payload["max_tokens"] = retry_max_tokens
                status, body = self._post_chat_completion(payload)
        if status >= 400:
            raise RuntimeError(
                f"OpenAI-compatible backend call failed: status={status} body={body[:2000]}"
            )

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenAI-compatible backend returned non-JSON: {body[:1000]}"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected OpenAI-compatible payload: {json.dumps(data, ensure_ascii=False)[:2000]}"
            ) from exc

        if not content:
            raise RuntimeError("OpenAI-compatible backend returned empty content.")
        return content


class QwenLocalClient:
    def __init__(
        self,
        model_dir: Path | str = "/workspace/meta-cognitive-RAG/models/Qwen2.5-32B-Instruct",
        max_output_tokens: int = 2048,
        max_input_tokens: int = 120000,
        per_gpu_max_memory_gib: int = 22,
    ) -> None:
        self.max_output_tokens = max_output_tokens
        self.backend_kind = _env_first(
            ("LAMBO_DAWONV4_LLM_BACKEND", "LAMBO_DAWON_LLM_BACKEND"),
            "transformers",
        ).lower()
        if self.backend_kind in {"server", "openai", "openai_compat", "vllm"}:
            self.config = None
            self.backend = _OpenAICompatChatBackend(
                base_url=_env_first(
                    ("LAMBO_DAWONV4_BASE_URL", "LAMBO_DAWON_BASE_URL"),
                    "http://127.0.0.1:1225/v1",
                ),
                model_name=_env_first(
                    ("LAMBO_DAWONV4_SERVER_MODEL_NAME", "LAMBO_DAWON_SERVER_MODEL_NAME"),
                    "Qwen",
                ),
                api_key=_env_first(
                    ("LAMBO_DAWONV4_API_KEY", "LAMBO_DAWON_API_KEY"),
                    "EMPTY",
                ),
                timeout=float(
                    _env_first(
                        ("LAMBO_DAWONV4_SERVER_TIMEOUT", "LAMBO_DAWON_SERVER_TIMEOUT"),
                        "1200",
                    )
                ),
            )
            self.backend_kind = "server"
        else:
            LocalTransformersBackend, LocalTransformersConfig = _load_local_backend()
            model_dir_override = _env_first(
                ("LAMBO_DAWONV4_MODEL_DIR", "LAMBO_DAWON_MODEL_DIR")
            )
            self.config = LocalTransformersConfig(
                model_id="Qwen/Qwen2.5-32B-Instruct",
                model_dir=Path(model_dir_override).expanduser() if model_dir_override else model_dir,
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
        output_tokens = max_output_tokens or self.max_output_tokens
        if self.backend_kind == "server":
            return self.backend.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=output_tokens,
            )
        return self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=output_tokens,
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
