from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import extract_json_payload


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class QwenLocalClient:
    def __init__(
        self,
        model_dir: Path | str | None = None,
        model_id: Optional[str] = None,
        max_output_tokens: int = 2048,
        max_input_tokens: int = 4096,
    ) -> None:
        configured_dir = model_dir or os.environ.get("LAMBO_MODEL_DIR") or os.environ.get("QWEN_MODEL_DIR")
        self.model_dir = Path(configured_dir) if configured_dir else None
        self.model_id = model_id or os.environ.get("LAMBO_MODEL_ID") or DEFAULT_MODEL_ID
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _model_ref(self) -> str:
        if self.model_dir and self.model_dir.exists():
            return str(self.model_dir)
        return self.model_id

    def _ensure_backend(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        if importlib.util.find_spec("accelerate") is None:
            raise RuntimeError(
                "The transformers backend is configured with device_map='auto', which requires the accelerate package. "
                "Install it in the same Python environment with: python -m pip install accelerate"
            )

        model_ref = self._model_ref()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize the transformers Qwen backend. "
                f"Tried model_ref={model_ref!r}. "
                "Set LAMBO_MODEL_ID to a Hugging Face model identifier, or optionally set LAMBO_MODEL_DIR to a local model path."
            ) from exc

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
        self._ensure_backend()
        assert self._model is not None and self._tokenizer is not None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        model_device = self._model.device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_output_tokens or self.max_output_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
