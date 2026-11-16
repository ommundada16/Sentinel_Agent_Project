"""Embedding providers (Strategy pattern, spec 5.8).

WHY a provider abstraction: tests stay offline/deterministic; the real model can be swapped in.
WHY same provider for docs and queries: mixing embedders silently ruins retrieval.
"""
import hashlib
import math
import re
from typing import Protocol

from app.data.models import EMBED_DIM


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic bag-of-words hashing embedder. Texts sharing words get high cosine similarity.
    Used for tests/offline demos; NOT semantic (no synonyms)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * EMBED_DIM
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % EMBED_DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class MiniLMEmbeddingProvider:
    """Real local model. Imported lazily so the heavy dependency stays optional."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def get_embedder(name: str | None = None) -> EmbeddingProvider:
    from app.config import get_settings

    name = name or get_settings().embedding_provider
    return MiniLMEmbeddingProvider() if name == "minilm" else HashEmbeddingProvider()
