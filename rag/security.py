"""Security helpers: input validation, upload sanitization, basic guards.

These are defensive measures for the app's own inputs — they make it much
harder for a malicious user to inject HTML/scripts, smuggle non-PDF files,
traverse the filesystem, or run up API costs.
"""

from __future__ import annotations

import hashlib
import html
import re
import time
from pathlib import Path

# ---- Limits (tune in one place) ----
MAX_FILE_MB = 25  # per uploaded file (browser uploads)
MAX_FILES_PER_UPLOAD = 30  # files accepted in a single upload batch
MAX_DOCS_PER_USER = 60  # cap on manually uploaded documents per user
# Drive scans read files in place (no browser upload), so they get a higher
# per-file cap and no per-user document cap — that's the whole point of them.
MAX_DRIVE_FILE_MB = 200
MAX_QUESTION_CHARS = 2000  # longest question we'll send to the model
RATE_LIMIT_MAX = 20  # max questions ...
RATE_LIMIT_WINDOW_S = 60  # ... per this many seconds, per session

# Strict email — only safe characters, so it can never carry HTML/script.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

# Everything outside this set is stripped from uploaded file names.
_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._\- ]")


def _sanitize_path_component(value: str) -> str:
    """Make one file-path component safe without changing its extension."""
    return _FILENAME_UNSAFE.sub("_", value).strip().strip(".")[:120]


def is_valid_email(email: str) -> bool:
    """True only for a normal, safe email address (no HTML-dangerous chars)."""
    email = (email or "").strip()
    return len(email) <= 254 and bool(_EMAIL_RE.match(email))


def esc(text: object) -> str:
    """HTML-escape a value before placing it inside `unsafe_allow_html` markup."""
    return html.escape(str(text), quote=True)


def sanitize_filename(name: str) -> str:
    """Strip path components and unsafe characters from an uploaded file name.

    Defeats path-traversal (``../../etc``) and odd characters, and guarantees a
    ``.pdf`` extension for display.
    """
    base = _sanitize_path_component(Path(name or "").name)  # drops traversal
    if not base:
        base = "document"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base


def sanitize_source_name(name: str) -> str:
    """Return a safe, readable relative path for citations.

    Directory uploads may provide names such as ``reports/2025/Q4.pdf``.
    Retaining those folder names makes otherwise-identical PDFs distinguishable
    in search results and source citations, while sanitising every component
    prevents a supplied path from escaping the application's storage area.
    """
    raw_parts = str(name or "").replace("\\", "/").split("/")
    parts = [
        _sanitize_path_component(part)
        for part in raw_parts
        if part not in {"", ".", ".."} and _sanitize_path_component(part)
    ]
    if not parts:
        return "document.pdf"
    parts[-1] = sanitize_filename(parts[-1])
    return "/".join(parts)


def looks_like_pdf(data: bytes) -> bool:
    """A genuine PDF begins with the ``%PDF`` signature near the start."""
    return data[:1024].lstrip()[:5].startswith(b"%PDF")


def storage_name(user_email: str, filename: str) -> str:
    """Collision-safe, traversal-safe on-disk name for an uploaded file.

    The user part is hashed (not the raw email) so the path can't be influenced
    by anything the user types, and it stays inside the upload directory.
    """
    user_digest = hashlib.sha1(user_email.lower().encode()).hexdigest()[:10]
    source = sanitize_source_name(filename)
    file_digest = hashlib.sha1(source.encode()).hexdigest()[:10]
    # Store only a safe basename on disk. The source path itself stays in
    # Chroma metadata, where it is used for citations.
    return f"{user_digest}__{file_digest}__{sanitize_filename(Path(source).name)}"


def rate_limited(history: list[float], now: float | None = None) -> bool:
    """Sliding-window rate limit. Mutates ``history`` in place; returns True if
    the caller has exceeded ``RATE_LIMIT_MAX`` requests in the window.

    ``now`` is injected (Streamlit passes ``time.time()``) so this stays testable.
    """
    now = time.time() if now is None else now
    cutoff = now - RATE_LIMIT_WINDOW_S
    history[:] = [t for t in history if t > cutoff]  # drop old entries
    if len(history) >= RATE_LIMIT_MAX:
        return True
    history.append(now)
    return False
