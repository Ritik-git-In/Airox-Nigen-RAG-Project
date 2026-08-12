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
- Write the answer as clean, natural prose, the way you would in a normal \
conversation. Do not include citation markers like [Source 2], footnotes, or a \
"Where I found this" list — the context is for grounding your answer, not for \
quoting back its labels.
- Be concise and precise. When numbers or dates matter, quote them exactly.
- Some excerpts end with a "Links on this page, top to bottom" list — those
are the real URLs behind hyperlinked text elsewhere on that same page (PDF
text extraction can't see a link's target, only its clickable label, so
they're supplied separately). When asked for links or references, match
each item to the link in the same position in that list and include the
actual URL, not just the label.
"""


@dataclass
class Answer:
    """An LLM answer plus the sources that were fed to it."""

    text: str
    sources: list[Retrieved]


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        api_key=config.KIMI_API_KEY,
        base_url=config.KIMI_BASE_URL,
        timeout=60.0,  # never leave the user staring at an endless spinner
        max_retries=2,
    )


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

    if not config.kimi_is_configured():
        return Answer(
            text=(
                "No Kimi API key is configured, so I can't compose an answer. "
                "The most relevant excerpts from your documents are listed "
                "below — set `KIMI_API_KEY` in `.env` to enable full answers."
            ),
            sources=chunks,
        )

    context = _format_context(chunks)
    user_prompt = (
        f"Context excerpts from the user's PDFs:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, as plain natural-language prose "
        "with no citation markers."
    )

    try:
        response = _client().chat.completions.create(
            model=config.KIMI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # low — we want faithful, grounded answers
            # Without an explicit cap, the API falls back to its own default
            # max_tokens, which is small enough that a detailed, multi-source
            # answer gets cut off mid-sentence with no error or warning.
            max_tokens=4096,
        )
    except Exception as exc:
        # Surface a friendly message (with the excerpts we found) instead of
        # letting a network/API error crash the app with a raw traceback.
        return Answer(
            text=(
                "Sorry — the language model request failed "
                f"({type(exc).__name__}). Please try again in a moment."
            ),
            sources=chunks,
        )
    return Answer(text=response.choices[0].message.content or "", sources=chunks)
