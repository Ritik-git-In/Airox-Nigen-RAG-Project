"""Standalone bulk PDF ingestion for large (multi-terabyte) drives.

Deliberately decoupled from the Streamlit app: a scan over a multi-terabyte
drive can take hours to days, and running it inside a Streamlit session ties
its survival to the browser tab staying open and the Streamlit process never
restarting. Run this from a terminal instead — it keeps going independently,
and the Streamlit app can be used normally at the same time to ask questions
about whatever has been indexed so far.

Parallel where it's safe, serial where it has to be: extracting text (and
running OCR on scanned pages) is CPU-heavy and has no shared state, so it
runs across a pool of worker processes. Embedding and writing to the vector
store and catalog all happen in this one main process — SQLite (which both
the catalog and Chroma's persistent client use) tolerates one writer well;
multiple processes writing to the same file at once would mostly just
serialize against each other anyway, with real risk of lock contention.

Safe to interrupt (Ctrl+C, power loss, crash) and re-run: every file's
result is written to the catalog the moment it's stored, so re-running the
exact same command later skips everything already done (matching size and
modified time) and only touches what's new or changed.

Usage:
    python scripts/bulk_index.py you@example.com E:\\
    python scripts/bulk_index.py you@example.com E:\\ --workers 8
    python scripts/bulk_index.py --list-drives
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import catalog, config, drive_scan, security, vectorstore  # noqa: E402
from rag.chunking import chunk_pages  # noqa: E402
from rag.pdf_loader import load_pdf  # noqa: E402


def _process_one_file(args: tuple[str, str, int, int]) -> dict:
    """Worker process entry point: extract + OCR + chunk one PDF.

    No database access here on purpose — this runs in a separate process,
    and keeping it side-effect-free (just CPU + this one file) means many
    of these can run at once with nothing to coordinate.
    """
    path_str, source, size, mtime = args
    try:
        pages = load_pdf(Path(path_str), source_name=source)
        chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        return {
            "ok": True,
            "source": source,
            "size": size,
            "mtime": mtime,
            "pages": len(pages),
            "chunks": chunks,
            "ocr_pages": sum(1 for p in pages if p.ocr),
        }
    except Exception as exc:  # a single bad file must not kill the worker
        return {
            "ok": False,
            "source": source,
            "size": size,
            "mtime": mtime,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _discover(root: str) -> list[Path]:
    print(f"Discovering PDFs under {root} ...", flush=True)
    stats: dict = {}
    pdfs: list[Path] = []
    last_update = 0.0
    for path in drive_scan.iter_pdfs(root, stats):
        pdfs.append(path)
        now = time.time()
        if now - last_update > 2.0:
            print(
                f"  ...{len(pdfs):,} PDFs found so far "
                f"({stats.get('dirs', 0):,} folders searched)",
                flush=True,
            )
            last_update = now
    print(
        f"Discovery complete: {len(pdfs):,} PDF(s) in "
        f"{stats.get('dirs', 0):,} folders.",
        flush=True,
    )
    return pdfs


def _plan(
    user_email: str, root: str, pdfs: list[Path]
) -> tuple[list[tuple[str, str, int, int]], int, int, int]:
    """Split discovered files into "needs indexing" vs. skip-worthy, without
    holding a catalog connection open for the (possibly very long) indexing
    phase that follows.
    """
    max_bytes = security.MAX_DRIVE_FILE_MB * 1024 * 1024
    to_process: list[tuple[str, str, int, int]] = []
    skipped = too_big = stat_failed = 0
    with catalog.session(user_email) as cat:
        for path in pdfs:
            source = drive_scan.source_name_for(root, path)
            try:
                fstat = path.stat()
            except OSError:
                stat_failed += 1
                continue
            rec = cat.get(source)
            if (
                rec
                and rec.get("size") == fstat.st_size
                and rec.get("mtime") == int(fstat.st_mtime)
            ):
                skipped += 1
                continue
            if fstat.st_size > max_bytes:
                too_big += 1
                continue
            to_process.append((str(path), source, fstat.st_size, int(fstat.st_mtime)))
    return to_process, skipped, too_big, stat_failed


def run(user_email: str, root: str, workers: int) -> None:
    user_email = user_email.strip().lower()
    config.ensure_dirs()

    pdfs = _discover(root)
    if not pdfs:
        print("No PDF files found — nothing to do.")
        return

    to_process, skipped, too_big, stat_failed = _plan(user_email, root, pdfs)
    print(
        f"{len(to_process):,} file(s) need indexing "
        f"({skipped:,} already up to date, {too_big:,} over "
        f"{security.MAX_DRIVE_FILE_MB} MB, {stat_failed:,} unreadable).",
        flush=True,
    )
    if not to_process:
        return

    new = no_text = failed = ocr_used = 0
    done = 0
    start = time.time()
    with catalog.session(user_email) as cat, mp.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_process_one_file, to_process, chunksize=4):
            done += 1
            source = result["source"]
            if not result["ok"]:
                failed += 1
                print(f"[{done:,}/{len(to_process):,}] FAILED {source}: {result['error']}")
                continue

            chunks = result["chunks"]
            stored = vectorstore.replace_source_chunks(user_email, source, chunks) if chunks else 0
            # Zero-chunk files are recorded too, so a re-run skips them
            # instead of retrying a file that will never have text.
            cat.upsert(
                source,
                {
                    "origin": "drive",
                    "path": source,
                    "size": result["size"],
                    "mtime": result["mtime"],
                    "pages": result["pages"],
                    "chunks": stored,
                    "ocr": result["ocr_pages"] > 0,
                },
            )
            if stored == 0:
                no_text += 1
            else:
                new += 1
            if result["ocr_pages"]:
                ocr_used += 1

            if done % 25 == 0 or done == len(to_process):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                remaining = len(to_process) - done
                eta_min = (remaining / rate / 60) if rate > 0 else 0
                print(
                    f"[{done:,}/{len(to_process):,}] {rate:.2f} files/sec, "
                    f"ETA {eta_min:,.0f} min — last: {source}",
                    flush=True,
                )

    elapsed = time.time() - start
    print()
    print(f"Done in {elapsed / 60:.1f} minutes.")
    print(
        f"Newly indexed: {new:,} ({ocr_used:,} needed OCR) | "
        f"already up to date: {skipped:,} | no extractable text: {no_text:,} | "
        f"over size limit: {too_big:,} | failed: {failed:,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("user_email", nargs="?", help="Account to index into")
    parser.add_argument("drive_path", nargs="?", help='e.g. "E:\\" or a folder path')
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (mp.cpu_count() or 4) - 1),
        help="Parallel extraction/OCR workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--list-drives", action="store_true", help="List available drives and exit"
    )
    args = parser.parse_args()

    if args.list_drives:
        for d in drive_scan.list_drives():
            print(d.describe())
        return

    if not args.user_email or not args.drive_path:
        parser.error("user_email and drive_path are required (or use --list-drives)")

    if not security.is_valid_email(args.user_email):
        parser.error(f"'{args.user_email}' doesn't look like a valid email address")

    run(args.user_email, args.drive_path, args.workers)


if __name__ == "__main__":
    # Windows spawns fresh worker processes rather than forking, so this
    # guard is required — without it, each worker would re-run this whole
    # module (including re-parsing argv) on import.
    mp.freeze_support()
    main()
