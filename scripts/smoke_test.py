"""Quick end-to-end check of the retrieval path (no PDF, no API key needed).

Run from the project root:  python scripts/smoke_test.py

It feeds synthetic page text through chunk -> embed -> store -> query and prints
the retrieved sources, then cleans up the test user's collection. If a Kimi key
is configured it also tries one real answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't print ✓ / … etc. Force UTF-8
# so the script's output works everywhere.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Allow running as `python scripts/smoke_test.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config, vectorstore  # noqa: E402
from rag.chunking import chunk_pages  # noqa: E402
from rag.pdf_loader import PageText  # noqa: E402

TEST_USER = "smoke_test@example.com"


def main() -> None:
    pages = [
        PageText(
            source="bank_report.pdf",
            page=1,
            text=(
                "Acme Bank reported net profit of 4.2 billion dollars in fiscal "
                "year 2025, up 12 percent from the prior year. The increase was "
                "driven by higher net interest income and lower loan loss provisions."
            ),
        ),
        PageText(
            source="bank_report.pdf",
            page=2,
            text=(
                "Total assets grew to 310 billion dollars. The bank's tier 1 "
                "capital ratio stood at 14.8 percent at year end, comfortably "
                "above the regulatory minimum."
            ),
        ),
        PageText(
            source="market_outlook.pdf",
            page=1,
            text=(
                "Analysts expect interest rates to decline gradually through 2026, "
                "which could compress net interest margins across the banking sector."
            ),
        ),
    ]

    print("1) Chunking…")
    chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"   {len(chunks)} chunks created")

    print("2) Embedding + storing (downloads the model on first run)…")
    vectorstore.reset_user(TEST_USER)  # start clean
    stored = vectorstore.add_chunks(TEST_USER, chunks)
    print(f"   {stored} chunks stored in Chroma")

    print("3) Querying: 'What was Acme Bank's net profit?'")
    results = vectorstore.query(TEST_USER, "What was Acme Bank's net profit?", top_k=3)
    for i, r in enumerate(results, start=1):
        print(f"   [{i}] {r.source} p{r.page} (distance={r.distance:.3f})")
        print(f"       {r.text[:90]}…")

    assert results, "Expected at least one retrieved chunk"
    assert results[0].source == "bank_report.pdf", "Top hit should be the profit page"
    print("   ✓ top hit is the profit page")

    if config.kimi_is_configured():
        print("4) Kimi key found — trying a real answer…")
        from rag import pipeline

        answer = pipeline.ask(TEST_USER, "What was Acme Bank's net profit?")
        print("   Answer:\n")
        print("   " + answer.text.replace("\n", "\n   "))
    else:
        print("4) No Kimi key configured — skipping the generation step.")

    vectorstore.reset_user(TEST_USER)  # clean up
    print("\nSmoke test passed ✅")


if __name__ == "__main__":
    main()
