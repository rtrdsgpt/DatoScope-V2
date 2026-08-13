"""
Retrieval over the co-pilot's RAG corpus (agent/rag_corpus.pkl, built by
scripts/06_build_rag_corpus.py from installed sklearn/scipy docstrings).

In-memory numpy cosine similarity — the corpus is a few dozen short chunks,
nowhere near large enough to need a real vector database.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "rag_corpus.pkl")


@dataclass
class RetrievedChunk:
    source: str
    topic: str
    text: str
    score: float


class RagRetriever:
    def __init__(self, corpus_path: str = CORPUS_PATH):
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"RAG corpus not found at {corpus_path}. Run: python scripts/06_build_rag_corpus.py"
            )
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)
        self.chunks = corpus["chunks"]
        self.embeddings = corpus["embeddings"]  # (n_chunks, dim), L2-normalized
        self.embedding_model_name = corpus["embedding_model"]
        self._model = None

    def _model_(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        query_vec = self._model_().encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec  # cosine similarity (both L2-normalized)
        top_idx = np.argsort(-scores)[:top_k]
        return [
            RetrievedChunk(source=self.chunks[i]["source"], topic=self.chunks[i]["topic"], text=self.chunks[i]["text"], score=float(scores[i]))
            for i in top_idx
        ]


@lru_cache(maxsize=1)
def get_retriever() -> RagRetriever:
    return RagRetriever()
