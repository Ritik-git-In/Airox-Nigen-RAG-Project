"""Orchestration: tie loading, chunking, embedding, retrieval, and answering.

This is the module the UI talks to. It hides the individual steps behind two
verbs: ``ingest_pdf`` (index a document) and ``ask`` (answer a question).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config, vectorstore
from .chunking import chunk_pages
from .llm import Answer, answer_question
from .pdf_loader import load_pdf


@dataclass
class IngestResult:
    """Summary of indexing a single PDF."""

    source: str
    pages: int
    chunks: int


def ingest_pdf(
    user_email: str, path: str | Path, source_name: str | None = None
) -> IngestResult:
    """Read, chunk, embed, and store one PDF for a user."""
    pages = load_pdf(path, source_name=source_name)
    chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    stored = vectorstore.add_chunks(user_email, chunks)
    name = source_name or Path(path).name
    return IngestResult(source=name, pages=len(pages), chunks=stored)


def ask(user_email: str, question: str, top_k: int | None = None) -> Answer:
    """Retrieve relevant chunks for a user and ask Kimi to answer."""
    k = top_k or config.TOP_K
    chunks = vectorstore.query(user_email, question, k)
    return answer_question(question, chunks)
