"""Extract text from PDF files, one record per page.

Keeping page boundaries lets us cite the exact page a piece of information
came from later in the pipeline. Pages with little or no text in the PDF's
own text layer (scanned images) fall back to OCR instead of being skipped.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# A page with fewer extracted characters than this is treated as
# probably-scanned and sent through OCR — a real text page is almost always
# far above this even if it's short (a title page, a mostly-blank page).
_OCR_MIN_CHARS = 20

# 300 DPI is the standard sweet spot for OCR accuracy vs. render time/memory
# on a typical scanned document page; going much higher rarely improves
# recognition but multiplies the pixel count (and OCR time) considerably.
_OCR_DPI = 300

# Set once we know whether OCR's dependencies (PyMuPDF, pytesseract, and the
# Tesseract binary itself) are actually usable on this machine, so we don't
# retry the same failing import/subprocess call for every page of every PDF.
_ocr_status: bool | None = None


@dataclass
class PageText:
    """Text extracted from a single PDF page."""

    source: str  # original file name, e.g. "Q4_report.pdf"
    page: int  # 1-based page number
    text: str
    ocr: bool = False  # True if this page's text came from OCR, not the PDF's text layer


def ocr_available() -> bool:
    """Whether OCR can actually run: PyMuPDF + pytesseract are importable
    *and* the Tesseract binary itself is on PATH (pytesseract shells out to
    it — the Python packages installing cleanly doesn't guarantee that).
    Checked once per process and cached, since a missing dependency won't
    fix itself mid-run and re-probing per page would be wasted work across
    a scan touching millions of pages.
    """
    global _ocr_status
    if _ocr_status is None:
        try:
            import pymupdf  # noqa: F401
            import pytesseract

            pytesseract.get_tesseract_version()
            _ocr_status = True
        except Exception:
            _ocr_status = False
    return _ocr_status


def _ocr_page_image(pdf_doc, page_index: int):
    """Render one page (0-based) of an already-open PyMuPDF document to a
    PIL image, at a resolution OCR can actually read reliably.
    """
    from PIL import Image

    pix = pdf_doc[page_index].get_pixmap(dpi=_OCR_DPI)
    return Image.open(io.BytesIO(pix.tobytes("png")))


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

    Pages the PDF's own text layer has little or nothing for are OCR'd
    instead — a scanned book or report is exactly the kind of thing a
    multi-terabyte drive full of PDFs is likely to contain, and without
    this those pages would silently contribute nothing to the index.
    Pages that still have no usable text after that (truly blank pages, or
    OCR unavailable/failing) are skipped, same as before.
    ``source_name`` overrides the file name shown in citations (useful when
    the file on disk has a temp name).
    """
    path = Path(path)
    name = source_name or path.name
    reader = PdfReader(str(path))

    # Opened lazily — only if some page's native text is thin enough to need
    # it — and once per file rather than once per page, since re-parsing the
    # whole PDF structure for every scanned page in a multi-hundred-page
    # book would dominate the ingestion time on its own.
    fitz_doc = None
    try:
        pages: list[PageText] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                # A single malformed page shouldn't abort the whole document.
                text = ""
            text = text.strip()

            used_ocr = False
            if len(text) < _OCR_MIN_CHARS and ocr_available():
                if fitz_doc is None:
                    import pymupdf

                    try:
                        fitz_doc = pymupdf.open(str(path))
                    except Exception:
                        logger.warning("Could not open %s for OCR", path, exc_info=True)
                        fitz_doc = False  # sentinel: don't retry per page
                if fitz_doc:
                    try:
                        import pytesseract

                        image = _ocr_page_image(fitz_doc, index - 1)
                        ocr_text = pytesseract.image_to_string(image).strip()
                        if len(ocr_text) > len(text):
                            text = ocr_text
                            used_ocr = True
                    except Exception:
                        logger.warning(
                            "OCR failed on %s page %d", path, index, exc_info=True
                        )

            if not text:
                continue
            try:
                links = _page_links(page)
            except Exception:
                links = []
            if links:
                text += "\n\nLinks on this page, top to bottom:\n" + "\n".join(links)
            pages.append(PageText(source=name, page=index, text=text, ocr=used_ocr))
        return pages
    finally:
        if fitz_doc:
            fitz_doc.close()
