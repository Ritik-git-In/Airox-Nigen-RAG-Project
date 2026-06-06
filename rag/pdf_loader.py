"""Extract text from PDF files, one record per page.

Keeping page boundaries lets us cite the exact page a piece of information came
from later in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class PageText:
    """Text extracted from a single PDF page."""

    source: str  # original file name, e.g. "Q4_report.pdf"
    page: int  # 1-based page number
    text: str


def load_pdf(path: str | Path, source_name: str | None = None) -> list[PageText]:
    """Read a PDF and return its pages as :class:`PageText` records.

    Pages whose text extraction yields nothing (e.g. scanned images) are
    skipped — they would only add empty chunks. ``source_name`` overrides the
    file name shown in citations (useful when the file on disk has a temp name).
    """
    path = Path(path)
    name = source_name or path.name
    reader = PdfReader(str(path))

    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # A single malformed page shouldn't abort the whole document.
            text = ""
        text = text.strip()
        if text:
            pages.append(PageText(source=name, page=index, text=text))
    return pages
