"""ChromaDB wrapper: one persistent collection per user.

Each logged-in user (identified by email) gets their own collection so their
documents stay isolated. We supply our own embeddings rather than letting Chroma
embed, because we want full control over the local model used.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass
from functools import lru_cache

import chromadb

from . import config
from .chunking import Chunk
from .embeddings import embed_query, embed_texts

# The app warms the client up from a background thread while the user is still
# logging in; the lock stops a concurrent first call from creating two
# PersistentClients over the same SQLite files.
_client_lock = threading.Lock()


@dataclass
class Retrieved:
    """A chunk returned from a similarity search, with its distance score."""

    text: str
    source: str
    page: int
    distance: float


@lru_cache(maxsize=1)
def _build_client() -> chromadb.ClientAPI:
    config.ensure_dirs()
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _client() -> chromadb.ClientAPI:
    """Create the persistent client once and reuse it (thread-safely).

    Streamlit reruns the whole script on every interaction; without this cache
    a brand-new PersistentClient (with its own SQLite handle + HNSW load) was
    built on each rerun, which was the main source of the lag.
    """
    with _client_lock:
        return _build_client()


def warm_up() -> None:
    """Initialize the persistent client ahead of first use (app warm-up)."""
    _client()


def _collection_name(user_email: str) -> str:
    """Build a Chroma-safe collection name from an email.

    Chroma names must be 3-63 chars, alphanumeric/_/-, starting and ending
    alphanumeric. We slugify the email and append a short hash to avoid
    collisions between users whose emails slugify to the same string.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", user_email.lower()).strip("_")
    digest = hashlib.sha1(user_email.lower().encode()).hexdigest()[:8]
    name = f"u_{slug}_{digest}"
    return name[:63]


def _get_collection(user_email: str):
    return _client().get_or_create_collection(
        name=_collection_name(user_email),
        metadata={"hnsw:space": "cosine"},
    )


# Chroma rejects very large ``add`` batches (max ~5461 records), and a single
# text-heavy PDF can exceed that. Storing in slices also bounds the peak memory
# used while embedding.
_ADD_BATCH_SIZE = 500


def add_chunks(user_email: str, chunks: list[Chunk]) -> int:
    """Embed and store chunks for a user. Returns the number stored."""
    if not chunks:
        return 0
    collection = _get_collection(user_email)
    for start in range(0, len(chunks), _ADD_BATCH_SIZE):
        batch = chunks[start : start + _ADD_BATCH_SIZE]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embed_texts([c.text for c in batch]),
            metadatas=[
                {"source": c.source, "page": c.page, "chunk_index": c.chunk_index}
                for c in batch
            ],
        )
    return len(chunks)


def query(user_email: str, question: str, top_k: int) -> list[Retrieved]:
    """Return the ``top_k`` most similar chunks to ``question``."""
    collection = _get_collection(user_email)
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=min(top_k, collection.count()),
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    return [
        Retrieved(
            text=doc,
            source=meta.get("source", "unknown"),
            page=int(meta.get("page", 0)),
            distance=float(dist),
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def list_sources(user_email: str) -> list[str]:
    """Distinct file names a user has indexed (for showing 'your documents')."""
    collection = _get_collection(user_email)
    if collection.count() == 0:
        return []
    metas = collection.get(include=["metadatas"])["metadatas"]
    return sorted({m.get("source", "unknown") for m in metas})


def delete_source(user_email: str, source: str) -> None:
    """Remove every chunk that came from a given file."""
    _get_collection(user_email).delete(where={"source": source})


def reset_user(user_email: str) -> None:
    """Delete a user's entire collection (all their documents)."""
    try:
        _client().delete_collection(_collection_name(user_email))
    except Exception:
        pass  # nothing to delete
