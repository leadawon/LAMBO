"""Embedding abstraction for lambo_org retrieval baseline.

This is intentionally small and pluggable. By default it uses
``sentence-transformers``; if the dependency is not installed, calling
``encode`` raises a clear error message explaining how to install it.

No LLM is involved in building embeddings — this is the critical difference
from the original ``lambo.agents.anchor_agent`` pipeline which invoked an LLM
to generate ``anchor_title`` / ``summary`` / ``key_entities`` for each span.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class EmbeddingBackend:
    model_name: str

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - install-time guard
            raise ImportError(
                "lambo_org requires `sentence-transformers` for the retrieval "
                "baseline. Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32, copy=False)


def cosine_rank(query_vec: np.ndarray, matrix: np.ndarray) -> List[int]:
    """Rank rows of ``matrix`` by cosine similarity to ``query_vec``.

    Both inputs are assumed to be L2-normalized (the sentence-transformers
    backend normalizes by default), so cosine similarity reduces to a dot
    product.
    """
    if matrix.size == 0:
        return []
    scores = matrix @ query_vec
    return list(np.argsort(-scores))
