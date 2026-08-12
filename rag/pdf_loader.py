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


def _page_links(page) -> list[str]:
    """URIs from a page's link annotations, top-to-bottom reading order.

    Hyperlinked text (e.g. a reference list where "GitHub" or "Wikipedia" is
    the clickable label) stores its URL as a link annotation, not as part of
    the visible text — ``extract_text()`` alone never sees it, so a question
    like "give me the links" had nothing to answer with beyond the labels.
    PDF y-coordinates increase upward, so sorting by the annotation's lower
    y (Rect[1]) descending recovers reading order for a typical single-column
    list, which is enough to let the model line each link up with the
    reference above it.
    """
    annots = page.get("/Annots")
    if not annots:
        return []
    positioned: list[tuple[float, str]] = []
    for annot in annots:
        try:
            obj = annot.get_object()
            if obj.get("/Subtype") != "/Link":
                continue
            action = obj.get("/A")
            uri = action.get("/URI") if action else None
            rect = obj.get("/Rect")
            if uri and rect:
                positioned.append((float(rect[1]), str(uri)))
        except Exception:
            continue  # a malformed annotation shouldn't drop the rest
    positioned.sort(key=lambda item: item[0], reverse=True)
    return [uri for _, uri in positioned]


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
        if not text:
            continue
        try:
            links = _page_links(page)
        except Exception:
            links = []
        if links:
            text += "\n\nLinks on this page, top to bottom:\n" + "\n".join(links)
        pages.append(PageText(source=name, page=index, text=text))
    return pages
