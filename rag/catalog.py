"""Per-user document catalog: SQLite-backed, next to the vector store.

Why it exists: listing a user's documents straight from Chroma means fetching
every chunk's metadata, which grows with the corpus — a drive scan can index
millions of PDFs. The catalog answers "which documents does this user have?"
instantly, and it also remembers each file's size/mtime so a drive re-scan
can skip unchanged files instead of re-embedding terabytes.

SQLite instead of a single JSON file: a JSON catalog has to be re-read and
re-written *in full* on every save, so each checkpoint during a long scan
gets slower as the catalog grows — a real bottleneck once it holds hundreds
of thousands of records. SQLite gives an O(1) upsert per file regardless of
how large the catalog already is, and — critically for a scan that might run
for hours or days — every write is durably committed to disk immediately, so
a crash or power loss loses at most the one file that was in flight, not
however much had accumulated since the last periodic checkpoint.

A record looks like::

    {"origin": "drive" | "upload", "path": "...", "size": 123, "mtime": 456,
     "pages": 10, "chunks": 42, "ocr": False}

Records with ``chunks == 0`` (no extractable text, even after OCR) are kept
so re-scans skip them, but they are hidden from the visible source list.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from . import config

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    source TEXT PRIMARY KEY,
    origin TEXT,
    path TEXT,
    size INTEGER,
    mtime INTEGER,
    pages INTEGER,
    chunks INTEGER,
    ocr INTEGER DEFAULT 0
)
"""

# One catalog is only ever driven by one scan/upload flow at a time in this
# app, but guard connection creation anyway — cheap, and avoids surprises if
# that ever changes.
_lock = threading.Lock()


def _dir() -> Path:
    d = config.DATA_DIR / "catalog"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db_path(user_email: str) -> Path:
    digest = hashlib.sha1(user_email.lower().encode()).hexdigest()[:16]
    return _dir() / f"{digest}.sqlite3"


def _json_path(user_email: str) -> Path:
    """Where the old, pre-SQLite catalog lived — only read for migration."""
    digest = hashlib.sha1(user_email.lower().encode()).hexdigest()[:16]
    return _dir() / f"{digest}.json"


class CatalogSession:
    """A reusable connection for many sequential catalog operations.

    Opening a fresh SQLite connection for every single file would add
    needless overhead across a scan touching millions of files. Use this
    directly for a bulk scan (``with catalog.session(email) as cat: ...``);
    the plain module-level functions below open one of these per call, which
    is fine for occasional, one-off use.
    """

    def __init__(self, user_email: str):
        self.user_email = user_email
        self._conn = sqlite3.connect(
            _db_path(user_email), timeout=30, check_same_thread=False
        )
        # WAL mode lets the Streamlit UI read the catalog (e.g. the document
        # list) while a separate long-running scan process is writing to it.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()
        _migrate_from_json(self._conn, user_email)

    def get(self, source: str) -> dict | None:
        row = self._conn.execute(
            "SELECT origin, path, size, mtime, pages, chunks, ocr "
            "FROM documents WHERE source = ?",
            (source,),
        ).fetchone()
        if row is None:
            return None
        return {
            "origin": row[0],
            "path": row[1],
            "size": row[2],
            "mtime": row[3],
            "pages": row[4],
            "chunks": row[5],
            "ocr": bool(row[6]),
        }

    def upsert(self, source: str, record: dict) -> None:
        """Insert or update one document's record, committed immediately."""
        self._conn.execute(
            """
            INSERT INTO documents (source, origin, path, size, mtime, pages, chunks, ocr)
            VALUES (:source, :origin, :path, :size, :mtime, :pages, :chunks, :ocr)
            ON CONFLICT(source) DO UPDATE SET
                origin=excluded.origin, path=excluded.path, size=excluded.size,
                mtime=excluded.mtime, pages=excluded.pages, chunks=excluded.chunks,
                ocr=excluded.ocr
            """,
            {
                "source": source,
                "origin": record.get("origin", "upload"),
                "path": record.get("path"),
                "size": record.get("size", 0),
                "mtime": record.get("mtime", 0),
                "pages": record.get("pages", 0),
                "chunks": record.get("chunks", 0),
                "ocr": int(bool(record.get("ocr", False))),
            },
        )
        self._conn.commit()

    def remove(self, source: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE source = ?", (source,))
        self._conn.commit()

    def sources(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT source FROM documents WHERE chunks != 0 ORDER BY source"
        ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CatalogSession":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def session(user_email: str) -> CatalogSession:
    """Open a reusable session for a batch of operations (e.g. a scan)."""
    with _lock:
        return CatalogSession(user_email)


def _migrate_from_json(conn: sqlite3.Connection, user_email: str) -> None:
    """One-time import of the pre-SQLite JSON catalog, if one exists.

    Renames the old file with a ``.migrated`` suffix afterwards so this only
    ever runs once per account.
    """
    old = _json_path(user_email)
    if not old.exists():
        return
    try:
        data = json.loads(old.read_text(encoding="utf-8"))
    except Exception:
        old.rename(old.with_suffix(".json.migrated"))
        return
    for source, rec in data.items():
        conn.execute(
            """
            INSERT INTO documents (source, origin, path, size, mtime, pages, chunks, ocr)
            VALUES (:source, :origin, :path, :size, :mtime, :pages, :chunks, 0)
            ON CONFLICT(source) DO NOTHING
            """,
            {
                "source": source,
                "origin": rec.get("origin", "upload"),
                "path": rec.get("path"),
                "size": rec.get("size", 0),
                "mtime": rec.get("mtime", 0),
                "pages": rec.get("pages", 0),
                "chunks": rec.get("chunks", 0),
            },
        )
    conn.commit()
    old.rename(old.with_suffix(".json.migrated"))


def get(user_email: str, source: str) -> dict | None:
    with session(user_email) as cat:
        return cat.get(source)


def upsert(user_email: str, source: str, record: dict) -> None:
    with session(user_email) as cat:
        cat.upsert(source, record)


def sources(user_email: str) -> list[str]:
    """Sorted list of the user's citeable documents.

    Accounts created before any catalog existed are migrated once: their
    source names are read from the vector store and written to a fresh
    catalog, so this never has to scan Chroma again.
    """
    with session(user_email) as cat:
        rows = cat._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        if rows[0] == 0:
            from . import vectorstore

            for name in vectorstore.list_sources(user_email):
                cat.upsert(name, {"origin": "upload", "chunks": 1})
        return cat.sources()


def remove(user_email: str, source: str) -> None:
    with session(user_email) as cat:
        cat.remove(source)


def delete_all(user_email: str) -> None:
    try:
        db = _db_path(user_email)
        db.unlink(missing_ok=True)
        Path(str(db) + "-wal").unlink(missing_ok=True)
        Path(str(db) + "-shm").unlink(missing_ok=True)
    except OSError:
        pass
