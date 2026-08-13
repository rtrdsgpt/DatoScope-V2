"""
06_build_rag_corpus.py
DatoScope — Step 6: Build the co-pilot's RAG corpus

Extracts docstrings directly from the installed sklearn/scipy versions
(pinned in requirements.txt) for the specific classes/functions DatoScope's
preprocessing, modeling, and EDA code actually uses, embeds each with a
local sentence-transformers model, and saves the corpus for the co-pilot's
retriever (agent/rag.py) to load. Always in sync with what's actually
installed — no scraping, no network fetch of arbitrary doc pages.

Run whenever CORPUS_SYMBOLS changes or the pinned sklearn/scipy version
changes:  python scripts/06_build_rag_corpus.py
"""

from __future__ import annotations

import inspect
import os
import pickle
import re

import numpy as np

CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent", "rag_corpus.pkl")

# (import path, symbol name, one-line topic tag) — the topic tag helps the
# co-pilot's prompt cite *why* a source is relevant, not just what it says.
CORPUS_SYMBOLS = [
    ("sklearn.preprocessing", "StandardScaler", "scaling"),
    ("sklearn.preprocessing", "MinMaxScaler", "scaling"),
    ("sklearn.preprocessing", "RobustScaler", "scaling"),
    ("sklearn.impute", "SimpleImputer", "missing values"),
    ("sklearn.linear_model", "LinearRegression", "regression"),
    ("sklearn.linear_model", "Ridge", "regression"),
    ("sklearn.linear_model", "Lasso", "regression"),
    ("sklearn.linear_model", "LogisticRegression", "classification"),
    ("sklearn.ensemble", "RandomForestClassifier", "classification"),
    ("sklearn.neighbors", "KNeighborsClassifier", "classification"),
    ("sklearn.cluster", "KMeans", "clustering"),
    ("sklearn.cluster", "DBSCAN", "clustering"),
    ("sklearn.cluster", "AgglomerativeClustering", "clustering"),
    ("sklearn.decomposition", "PCA", "dimensionality reduction"),
    ("sklearn.metrics", "r2_score", "regression metric"),
    ("sklearn.metrics", "mean_squared_error", "regression metric"),
    ("sklearn.metrics", "mean_absolute_error", "regression metric"),
    ("sklearn.metrics", "accuracy_score", "classification metric"),
    ("sklearn.metrics", "precision_score", "classification metric"),
    ("sklearn.metrics", "recall_score", "classification metric"),
    ("sklearn.metrics", "f1_score", "classification metric"),
    ("sklearn.metrics", "silhouette_score", "clustering metric"),
    ("sklearn.metrics", "davies_bouldin_score", "clustering metric"),
    ("sklearn.metrics", "calinski_harabasz_score", "clustering metric"),
    ("sklearn.metrics", "fowlkes_mallows_score", "clustering metric"),
    ("sklearn.metrics", "rand_score", "clustering metric"),
    ("sklearn.model_selection", "cross_val_score", "model validation"),
    ("sklearn.model_selection", "train_test_split", "model validation"),
    ("scipy.stats", "zscore", "outlier detection"),
    ("scipy.stats", "iqr", "outlier detection"),
    ("scipy.stats", "skew", "distribution shape"),
    ("scipy.stats", "kurtosis", "distribution shape"),
    ("scipy.stats", "probplot", "distribution shape"),
]


def _summary(doc: str, max_chars: int = 1200) -> str:
    """The first paragraph(s) of a numpydoc-style docstring, before the
    Parameters/Attributes section — the plain-language description, not the
    full parameter reference (keeps chunks focused for retrieval and short
    enough to fit several in a prompt)."""
    doc = inspect.cleandoc(doc)
    cut = re.search(r"\n\s*(Parameters|Attributes)\s*\n\s*-{3,}", doc)
    body = doc[: cut.start()] if cut else doc
    body = re.sub(r"\n{2,}", "\n\n", body).strip()
    return body[:max_chars]


def build() -> None:
    import importlib

    from sentence_transformers import SentenceTransformer

    print("Loading sentence-transformers/all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = []
    for module_path, symbol, topic in CORPUS_SYMBOLS:
        module = importlib.import_module(module_path)
        obj = getattr(module, symbol)
        doc = inspect.getdoc(obj)
        if not doc:
            print(f"  skip {module_path}.{symbol} (no docstring)")
            continue
        text = _summary(doc)
        chunks.append({"source": f"{module_path}.{symbol}", "topic": topic, "text": text})
        print(f"  extracted {module_path}.{symbol} ({len(text)} chars)")

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

    corpus = {
        "chunks": chunks,
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "embedding_model": "all-MiniLM-L6-v2",
    }
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    with open(CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f)
    print(f"Saved {len(chunks)} chunks -> {CORPUS_PATH}")


if __name__ == "__main__":
    build()
