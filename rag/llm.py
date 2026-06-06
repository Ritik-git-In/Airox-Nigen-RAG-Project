"""Kimi / Moonshot AI client (OpenAI-compatible) for the generation step.

Moonshot exposes an OpenAI-compatible Chat Completions API, so we use the
official ``openai`` SDK and just point ``base_url`` at Moonshot's endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from . import config
from .vectorstore import Retrieved

SYSTEM_PROMPT = """You are a careful research assistant that answers questions \
using ONLY the provided document excerpts (the "context"). The documents are \
PDFs the user uploaded (e.g. financial reports, filings).

Rules:
- Base your answer strictly on the context. If the context does not contain the \
answer, say so plainly — do not invent facts.
- Cite your sources inline using the bracket labels shown in the context, like \
[Source 2]. Every claim that relies on a document must carry at least one citation.
- Be concise and precise. When numbers or dates matter, quote them exactly.
- At the end, add a short "Where I found this" list mapping each [Source N] you \
used to its file name and page number.
"""


@dataclass
class Answer:
    """An LLM answer plus the sources that were fed to it."""

    text: str
    sources: list[Retrieved]


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=config.KIMI_API_KEY, base_url=config.KIMI_BASE_URL)


def _format_context(chunks: list[Retrieved]) -> str:
    """Render retrieved chunks as labelled blocks the model can cite."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = f"[Source {i}] (file: {c.source}, page: {c.page})"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def answer_question(question: str, chunks: list[Retrieved]) -> Answer:
    """Ask Kimi to answer ``question`` grounded in the retrieved ``chunks``."""
    if not chunks:
        return Answer(
            text=(
                "I couldn't find anything relevant in your uploaded documents. "
                "Try rephrasing the question, or upload PDFs that cover this topic."
            ),
            sources=[],
        )

    context = _format_context(chunks)
    user_prompt = (
        f"Context excerpts from the user's PDFs:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with inline [Source N] citations."
    )

    response = _client().chat.completions.create(
        model=config.KIMI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low — we want faithful, grounded answers
    )
    return Answer(text=response.choices[0].message.content or "", sources=chunks)
