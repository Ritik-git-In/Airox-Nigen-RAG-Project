"""Local text embeddings via ChromaDB's bundled ONNX MiniLM model.

We use the same ``all-MiniLM-L6-v2`` model, but through ``onnxruntime`` instead
of PyTorch. That is dramatically lighter on RAM and disk — which is what lets
the app run on small free-tier instances (e.g. Render's 512 MB). The model
(~80 MB) downloads automatically on first use and is cached after that.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from chromadb.utils import embedding_functions

# The app warms the model up from a background thread while the user is still
# logging in; the lock stops a concurrent first call from loading it twice.
_embedder_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_embedder():
    return embedding_functions.ONNXMiniLM_L6_V2()


def _embedder():
    """Load (and memoize) the ONNX embedding function, thread-safely."""
    with _embedder_lock:
        return _load_embedder()


def warm_up() -> None:
    """Download/load the model and embed once so the first real call is fast."""
    embed_texts(["warm-up"])


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into vectors (384-dim, suitable for cosine search)."""
    if not texts:
        return []
    vectors = _embedder()(texts)
    return [list(map(float, v)) for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
