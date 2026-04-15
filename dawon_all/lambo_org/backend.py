"""Standalone LLM backend for lambo_org.

The original ``lambo.backend`` imports ``meta_cognitive_rag.local_backend``
as an external wrapper around ``transformers``. That dependency is not
always available, but it is really just a thin model-loading layer.

To keep ``lambo_org`` self-contained and usable with any local Qwen
checkpoint, we implement the same ``generate_text`` / ``generate_json``
interface directly on top of ``transformers``. No external service, no
API key, no ``meta_cognitive_rag`` package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

from lambo.common import extract_json_payload


DEFAULT_MODEL_CANDIDATES = [
    os.environ.get("LAMBO_ORG_QWEN_MODEL_DIR", ""),
    "/workspace/StructRAG/model/Qwen2.5-32B-Instruct",
    "/workspace/qwen/Qwen2.5-32B-Instruct",
    "/workspace/meta-cognitive-RAG/models/Qwen2.5-32B-Instruct",
]


def _resolve_model_dir(explicit: Optional[str] = None) -> str:
    candidates = [explicit] if explicit else []
    candidates.extend(DEFAULT_MODEL_CANDIDATES)
    for path in candidates:
        if path and Path(path).is_dir():
            return str(path)
    raise FileNotFoundError(
        "No local Qwen model directory found. Set LAMBO_ORG_QWEN_MODEL_DIR "
        "or place weights at /workspace/StructRAG/model/Qwen2.5-32B-Instruct."
    )


class QwenLocalClient:
    """Local transformers-backed Qwen client (no external packages)."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        max_output_tokens: int = 2048,
        max_input_tokens: int = 32768,
        dtype: str = "float16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = _resolve_model_dir(model_dir)
        self.model_dir = resolved
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
            dtype, torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            resolved,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def _build_input_ids(self, system_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in enc.items()}

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> str:
        import torch

        inputs = self._build_input_ids(system_prompt, user_prompt)
        input_len = inputs["input_ids"].shape[1]
        gen_kwargs = {
            "max_new_tokens": max_output_tokens or self.max_output_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> Tuple[Any, str]:
        raw = self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            metadata=metadata,
        )
        payload = extract_json_payload(raw)
        if payload is None:
            repair = (
                user_prompt
                + "\n\nYour previous answer was not valid JSON. Re-answer "
                "with strict JSON only. No markdown fences, no commentary."
            )
            raw = self.generate_text(
                system_prompt=system_prompt,
                user_prompt=repair,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                metadata=metadata,
            )
            payload = extract_json_payload(raw)
        return payload, raw


def get_default_client(backend: str = "local"):
    if backend == "local":
        return QwenLocalClient()
    if backend == "gemini":
        # Delegate to the original lambo.backend so GEMINI_API_KEY path is reused.
        from lambo.backend import GeminiClient  # type: ignore

        return GeminiClient()
    raise ValueError(f"Unknown backend: {backend}")
