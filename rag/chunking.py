"""Split page text into overlapping chunks suitable for embedding.

Why chunk at all? Embedding a whole page (or document) into one vector blurs
many topics together, so retrieval gets imprecise. Smaller overlapping chunks
give sharper matches, and the overlap keeps sentences that straddle a boundary
from being cut in half.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pdf_loader import PageText


@dataclass
class Chunk:
    """A slice of text plus the metadata needed to cite it."""

    id: str  # unique within a user's collection
    text: str
    source: str  # file name
    page: int  # page number
    chunk_index: int  # position of this chunk within its page


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedy character-window split with overlap.

    We try to break on a whitespace boundary near the window edge so we don't
    slice through the middle of a word.
    """
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        overlap = chunk_size // 4  # guard against a non-advancing window

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # Prefer to end on a space if we're not already at the document end.
        if end < n:
            space = text.rfind(" ", start, end)
            if space != -1 and space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(
    pages: list[PageText], chunk_size: int, overlap: int
) -> list[Chunk]:
    """Turn page records into citeable :class:`Chunk` objects."""
    chunks: list[Chunk] = []
    for page in pages:
        for i, piece in enumerate(_split_text(page.text, chunk_size, overlap)):
            chunk_id = f"{page.source}::p{page.page}::c{i}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    source=page.source,
                    page=page.page,
                    chunk_index=i,
                )
            )
    return chunks
