"""Embedding providers.

Must produce the SAME vectors as Demo 1's ingestion, otherwise the query vector
and the stored chunk vectors live in different spaces and retrieval is garbage.
"""
from typing import List

from fastembed import TextEmbedding
from openai import OpenAI

import config


class FastEmbedEmbedder:
    """Local, offline embeddings via the `fastembed` library (ONNX runtime)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)
        probe = next(iter(self.model.embed(["dimension probe"])))
        self._dim = len(probe)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self.model.embed(list(texts))]


class OpenAIEmbedder:
    """Embeddings via the OpenAI API (requires OPENAI_API_KEY)."""

    _KNOWN_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI()
        self._dim = self._KNOWN_DIMS.get(model_name)
        if self._dim is None:
            self._dim = len(self.encode(["dimension probe"])[0])

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model_name, input=list(texts))
        return [item.embedding for item in resp.data]


def get_embedder():
    """Build the embedder selected by EMBEDDING_PROVIDER in config/.env."""
    provider = config.EMBEDDING_PROVIDER
    if provider == "fastembed":
        return FastEmbedEmbedder(config.EMBEDDING_MODEL or "BAAI/bge-small-en-v1.5")
    if provider == "openai":
        return OpenAIEmbedder(config.EMBEDDING_MODEL or "text-embedding-3-small")
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={provider!r}. Use 'fastembed' or 'openai'."
    )
