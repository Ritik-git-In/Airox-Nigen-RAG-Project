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
    # Keep the digest intact. Truncating the final name can cut it off for a
    # long email address, allowing otherwise different accounts with the same
    # prefix to resolve to one collection.
    max_slug_len = 63 - len("u_") - len("_") - len(digest)
    return f"u_{slug[:max_slug_len]}_{digest}"


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
    return _add_chunks_to_collection(collection, chunks)


def _add_chunks_to_collection(collection, chunks: list[Chunk]) -> int:
    """Add chunks to an already selected collection in safe-sized batches."""
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


def replace_source_chunks(user_email: str, source: str, chunks: list[Chunk]) -> int:
    """Replace one source without losing its previous index if insertion fails.

    Re-uploads reuse stable chunk IDs.  The old code deleted those IDs before
    embedding the replacement, so an embedding/database failure left the user
    with no searchable copy of that document.  Keep a small Chroma snapshot
    and restore it if adding the replacement cannot complete.
    """
    collection = _get_collection(user_email)
    previous = collection.get(
        where={"source": source},
        include=["documents", "embeddings", "metadatas"],
    )
    collection.delete(where={"source": source})
    try:
        return _add_chunks_to_collection(collection, chunks) if chunks else 0
    except Exception:
        if previous["ids"]:
            collection.add(
                ids=previous["ids"],
                documents=previous["documents"],
                embeddings=previous["embeddings"],
                metadatas=previous["metadatas"],
            )
        raise


def query(user_email: str, question: str, top_k: int) -> list[Retrieved]:
    """Return the ``top_k`` most similar chunks to ``question``, plus each
    hit's immediate next chunk on the same page.

    A long list — a references section, a bullet list of features — often
    spans more than one 1000-character chunk. Without the neighbor, a "list
    everything in section X" question could match only the first half of
    that list (whichever chunk scored higher) and silently drop the rest.
    Pulling in chunk_index + 1 whenever chunk_index scores as a hit keeps
    split content whole. Neighbors are fetched by id (a direct lookup, not
    another similarity search) so this costs one extra cheap call, not a
    second embedding.
    """
    collection = _get_collection(user_email)
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=min(top_k, collection.count()),
    )
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    hits = [
        Retrieved(
            text=doc,
            source=meta.get("source", "unknown"),
            page=int(meta.get("page", 0)),
            distance=float(dist),
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]

    seen_ids = set(ids)
    neighbor_ids = []
    for meta in metas:
        neighbor_id = (
            f"{meta.get('source', 'unknown')}"
            f"::p{meta.get('page', 0)}::c{int(meta.get('chunk_index', 0)) + 1}"
        )
        if neighbor_id not in seen_ids:
            seen_ids.add(neighbor_id)
            neighbor_ids.append(neighbor_id)

    if neighbor_ids:
        neighbors = collection.get(ids=neighbor_ids, include=["documents", "metadatas"])
        for doc, meta in zip(neighbors["documents"], neighbors["metadatas"]):
            hits.append(
                Retrieved(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    page=int(meta.get("page", 0)),
                    distance=1.0,  # a neighbor pulled in for completeness, not independently ranked
                )
            )
    return hits


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
