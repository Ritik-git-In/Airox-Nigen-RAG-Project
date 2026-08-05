"""Discover connected drives and every PDF stored on them.

Indexing straight off the drive (instead of uploading through the browser) is
what makes terabyte-scale collections practical: nothing is copied, files are
read in place, and a re-scan only touches files whose size/mtime changed since
the last run (tracked in :mod:`rag.catalog`). This only works when the app
runs on the same machine the drive is plugged into — which is exactly the
self-hosted setup this app targets.
"""

from __future__ import annotations

import os
import shutil
import string
from dataclasses import dataclass
from pathlib import Path

from . import security

# Junk/system folders that can't contain user documents. Skipping them avoids
# permission-error noise, recycle-bin duplicates, and wasted walk time.
_SKIP_DIRS = {
    "$RECYCLE.BIN",
    "SYSTEM VOLUME INFORMATION",
    "$WINDOWS.~BT",
    "$WINREAGENT",
    "WINDOWS.OLD",
    "RECOVERY",
    "MSOCACHE",
}

_DRIVE_KINDS = {
    2: "Removable",
    3: "Local disk",
    4: "Network",
    5: "CD/DVD",
    6: "RAM disk",
}


@dataclass(frozen=True)
class DriveInfo:
    """One mounted drive the user can pick in the UI."""

    root: str  # e.g. "E:\\"
    label: str  # volume label, may be empty
    kind: str  # "Removable", "Local disk", ...
    total: int  # bytes
    free: int  # bytes

    def describe(self) -> str:
        label = f" {self.label}" if self.label else ""
        return (
            f"{self.root}{label} ({self.kind}) — "
            f"{_human(self.free)} free of {_human(self.total)}"
        )


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PB"


def list_drives() -> list[DriveInfo]:
    """Every ready drive on this machine (USB, external, local, network)."""
    if os.name != "nt":  # non-Windows fallback: offer the filesystem root
        usage = shutil.disk_usage("/")
        return [DriveInfo("/", "", "Local disk", usage.total, usage.free)]

    import ctypes

    kernel32 = ctypes.windll.kernel32
    drives: list[DriveInfo] = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if not os.path.exists(root):
            continue  # not mounted / no media
        kind = _DRIVE_KINDS.get(
            kernel32.GetDriveTypeW(ctypes.c_wchar_p(root)), "Drive"
        )
        buf = ctypes.create_unicode_buffer(261)
        try:
            kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(root), buf, len(buf), None, None, None, None, 0
            )
        except Exception:
            pass  # label stays empty
        try:
            usage = shutil.disk_usage(root)
        except OSError:
            continue  # drive vanished between exists() and here
        drives.append(DriveInfo(root, buf.value, kind, usage.total, usage.free))
    return drives


def _extended(path: str) -> str:
    """Return the ``\\\\?\\`` extended-length form of an absolute Windows path.

    Classic Win32 file APIs fail on paths longer than ~260 characters, which
    deep folder trees on multi-terabyte drives routinely exceed. Walking with
    the extended prefix lifts that limit for every downstream operation too
    (stat, open, pypdf), because yielded paths inherit the prefix.
    """
    if os.name == "nt" and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(path)
    return path


def strip_extended(path: str | Path) -> str:
    """Undo :func:`_extended` for display and citation naming."""
    s = str(path)
    return s[4:] if s.startswith("\\\\?\\") else s


def iter_pdfs(root: str | Path, stats: dict | None = None):
    """Yield the path of every ``.pdf`` under ``root``, lazily.

    Unreadable folders are skipped silently (external drives often carry
    permission oddities), junk/system folders are pruned, and if ``stats`` is
    given its ``"dirs"`` counter is updated so the UI can show live progress
    while multi-terabyte trees are being walked.
    """
    for dirpath, dirnames, filenames in os.walk(
        _extended(str(root)), onerror=lambda _e: None
    ):
        dirnames[:] = [d for d in dirnames if d.upper() not in _SKIP_DIRS]
        if stats is not None:
            stats["dirs"] = stats.get("dirs", 0) + 1
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                yield Path(dirpath) / fn


def source_name_for(root: str | Path, path: Path) -> str:
    """Citeable, sanitized name like ``E/reports/2025/q4.pdf``.

    Keeping the drive letter and folder structure makes otherwise
    identically-named PDFs distinguishable in citations, and gives a stable
    identity so re-scans update the same document instead of duplicating it.
    """
    root_s = strip_extended(os.path.abspath(str(root)))
    path_s = strip_extended(path)
    drive = Path(root_s).drive.rstrip(":") or Path(root_s).anchor.strip("/\\") or "drive"
    try:
        rel = Path(path_s).relative_to(root_s).as_posix()
    except ValueError:
        rel = Path(path_s).name
    return security.sanitize_source_name(f"{drive}/{rel}")
