"""Per-user document catalog: a small JSON sidecar next to the vector store.

Why it exists: listing a user's documents straight from Chroma means fetching
every chunk's metadata, which grows with the corpus — a drive scan can index
tens of thousands of PDFs. The catalog answers "which documents does this user
have?" instantly, and it also remembers each file's size/mtime so a drive
re-scan can skip unchanged files instead of re-embedding terabytes.

A record looks like::

    {"origin": "drive" | "upload", "path": "...", "size": 123, "mtime": 456,
     "pages": 10, "chunks": 42}

Records with ``chunks == 0`` (scanned images with no extractable text) are kept
so re-scans skip them, but they are hidden from the visible source list.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config


def _file(user_email: str) -> Path:
    digest = hashlib.sha1(user_email.lower().encode()).hexdigest()[:16]
    return config.DATA_DIR / "catalog" / f"{digest}.json"


def load(user_email: str) -> dict[str, dict] | None:
    """The user's catalog, or ``None`` if none has been written yet.

    ``None`` (no file) is distinct from ``{}`` (an intentionally empty
    catalog): the former triggers a one-time migration from the vector store
    for accounts that predate the catalog.
    """
    f = _file(user_email)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None  # corrupt file -> rebuild via migration rather than crash


def save(user_email: str, cat: dict[str, dict]) -> None:
    f = _file(user_email)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cat), encoding="utf-8")
    tmp.replace(f)  # atomic-ish replace so a crash can't truncate the catalog


def sources(user_email: str) -> list[str]:
    """Sorted list of the user's citeable documents.

    Accounts created before the catalog existed are migrated once: their
    source names are read from the vector store and written to a fresh
    catalog, so this never has to scan Chroma again.
    """
    cat = load(user_email)
    if cat is None:
        from . import vectorstore

        cat = {
            name: {"origin": "upload", "chunks": 1, "seeded": True}
            for name in vectorstore.list_sources(user_email)
        }
        save(user_email, cat)
    return sorted(s for s, rec in cat.items() if rec.get("chunks", 0) != 0)


def remove(user_email: str, source: str) -> None:
    cat = load(user_email) or {}
    if source in cat:
        del cat[source]
        save(user_email, cat)


def delete_all(user_email: str) -> None:
    try:
        _file(user_email).unlink(missing_ok=True)
    except OSError:
        pass
