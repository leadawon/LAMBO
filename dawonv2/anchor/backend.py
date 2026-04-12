from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import requests
import torch

from .common import extract_json_payload
from .paths import resolve_local_model_dir

_TRANSFORMERS_IMPORT_ERROR: Exception | None = None
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc


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
        self.session = requests.Session()
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
        margin = int(os.environ.get("LAMBO_DAWON_CONTEXT_RETRY_MARGIN", "64"))
        available = model_limit - prompt_tokens - max(0, margin)
        if available < 1:
            available = model_limit - prompt_tokens
        if 0 < available < requested:
            return available
        return None

    def _post_chat_completion(self, payload: dict[str, Any]) -> requests.Response:
        return self.session.post(
            self.chat_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )

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
        response = self._post_chat_completion(payload)
        if response.status_code == 400:
            retry_max_tokens = self._context_retry_max_tokens(response.text, max_output_tokens)
            if retry_max_tokens is not None:
                payload["max_tokens"] = retry_max_tokens
                response = self._post_chat_completion(payload)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            snippet = response.text[:2000]
            raise RuntimeError(
                f"OpenAI-compatible backend call failed: status={response.status_code} body={snippet}"
            ) from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenAI-compatible backend returned non-JSON: {response.text[:1000]}"
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


class _TransformersChatBackend:
    """Thin local-inference wrapper around a Hugging Face causal LM."""

    def __init__(
        self,
        *,
        model_dir: Path,
        max_output_tokens: int,
        per_gpu_max_memory_gib: int,
    ) -> None:
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is not available. Run with a Python environment that has "
                "`transformers`, and for multi-GPU auto-sharding also install `accelerate`. "
                "Recommended: /workspace/venvs/ragteamvenv/bin/python."
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.model_dir = model_dir
        self.max_output_tokens = max_output_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {
                gpu_index: f"{per_gpu_max_memory_gib}GiB"
                for gpu_index in range(torch.cuda.device_count())
            }

        self.model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_device = getattr(self.model, "device", None)
        if model_device is None:
            model_device = next(self.model.parameters()).device
        inputs = self.tokenizer([prompt], return_tensors="pt").to(model_device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class QwenLocalClient:
    def __init__(
        self,
        model_dir: Path | str | None = None,
        max_output_tokens: int = 2048,
        max_input_tokens: int = 120000,
        per_gpu_max_memory_gib: int = 22,
    ) -> None:
        del max_input_tokens
        resolved_model_dir = Path(model_dir) if model_dir is not None else resolve_local_model_dir(strict=True)
        self.max_output_tokens = max_output_tokens
        self.model_dir = resolved_model_dir
        self.backend_kind = os.environ.get("LAMBO_DAWON_LLM_BACKEND", "transformers").strip().lower()

        if self.backend_kind in {"server", "openai", "openai_compat", "vllm"}:
            self.backend = _OpenAICompatChatBackend(
                base_url=os.environ.get("LAMBO_DAWON_BASE_URL", "http://127.0.0.1:1225/v1"),
                model_name=os.environ.get("LAMBO_DAWON_SERVER_MODEL_NAME", "Qwen"),
                api_key=os.environ.get("LAMBO_DAWON_API_KEY", "EMPTY"),
                timeout=float(os.environ.get("LAMBO_DAWON_SERVER_TIMEOUT", "1200")),
            )
            self.backend_kind = "server"
        else:
            self.backend = _TransformersChatBackend(
                model_dir=resolved_model_dir,
                max_output_tokens=max_output_tokens,
                per_gpu_max_memory_gib=per_gpu_max_memory_gib,
            )
            self.backend_kind = "transformers"

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        del metadata
        return self.backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens or self.max_output_tokens,
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


@lru_cache(maxsize=1)
def get_default_client() -> QwenLocalClient:
    return QwenLocalClient()
