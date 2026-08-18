"""Orchestration: tie loading, chunking, embedding, retrieval, and answering.

This is the module the UI talks to. It hides the individual steps behind two
verbs: ``ingest_pdf`` (index a document) and ``ask`` (answer a question).
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from pathlib import Path

from . import config, vectorstore
from .chunking import chunk_pages
from .llm import Answer, answer_question
from .pdf_loader import load_pdf

# Retrieval is normally near-instant (a local ONNX embed + a local Chroma
# query, no network). It has no built-in timeout of its own, though, so if
# it ever gets stuck — e.g. a locked SQLite file from another process still
# holding the same collection open — a question would otherwise hang the UI
# forever with no error and no way out short of restarting the server. This
# bounds that wait so the user gets a clear error message instead.
_RETRIEVAL_TIMEOUT_S = 30
_retrieval_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="retrieval"
)


@dataclass
class IngestResult:
    """Summary of indexing a single PDF."""

    source: str
    pages: int
    chunks: int
    ocr_pages: int = 0  # how many of those pages needed OCR (scanned/image pages)


def ingest_pdf(
    user_email: str, path: str | Path, source_name: str | None = None
) -> IngestResult:
    """Read, chunk, embed, and store one PDF for a user."""
    pages = load_pdf(path, source_name=source_name)
    chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    name = source_name or Path(path).name
    # Re-uploading refreshes the source while preserving the prior index if
    # embedding or storage fails partway through the replacement.
    stored = vectorstore.replace_source_chunks(user_email, name, chunks)
    ocr_pages = sum(1 for p in pages if p.ocr)
    return IngestResult(source=name, pages=len(pages), chunks=stored, ocr_pages=ocr_pages)


def ask(user_email: str, question: str, top_k: int | None = None) -> Answer:
    """Retrieve relevant chunks for a user and ask Kimi to answer."""
    k = top_k or config.TOP_K
    future = _retrieval_pool.submit(vectorstore.query, user_email, question, k)
    try:
        chunks = future.result(timeout=_RETRIEVAL_TIMEOUT_S)
    except concurrent.futures.TimeoutError:
        return Answer(
            text=(
                "Sorry — searching your documents is taking much longer than "
                "usual and timed out. This can happen if another process has "
                "the document index locked; try again in a moment, and "
                "restart the app if it keeps happening."
            ),
            sources=[],
        )
    return answer_question(question, chunks)
