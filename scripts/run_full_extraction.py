"""
Full PDF Extraction Driver — parallel extraction over a local PDF tree.

Walks PDF_ROOT (default: data/pdfs/) for .pdf files, extracts + chunks each
in parallel via ProcessPoolExecutor, writes data/chunks/all_chunks.json plus
a manifest.

Why local: the original Mar 2026 Colab session ran on Colab's local fast disk.
Running directly against Google Drive Desktop's CloudStorage FUSE mount is
~10x slower due to per-syscall FUSE overhead. Best to rsync PDFs locally
first (~10 min) then extract on local SSD (~5 min).

Usage:
    python3 scripts/run_full_extraction.py
"""

import os
import sys
import json
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Reuse the canonical chunk/extract logic from extract_and_chunk.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from extract_and_chunk import process_single_pdf, find_all_pdfs  # noqa: E402

PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PDF_ROOT = os.environ.get("PDF_ROOT", os.path.join(PROJECT_DIR, "data", "pdfs"))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "chunks")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "6"))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "100"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info(f"Scanning PDFs under {PDF_ROOT}")
    t0 = time.time()
    pdfs = find_all_pdfs(PDF_ROOT)
    log.info(f"  Found {len(pdfs)} PDFs in {time.time()-t0:.1f}s")
    if not pdfs:
        log.error("No PDFs found. Check PDF_ROOT.")
        return

    args_list = [(p, PDF_ROOT) for p in pdfs]

    all_chunks = []
    processed = 0
    failed = 0
    failed_paths = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_single_pdf, a): a[0] for a in args_list}
        for fut in as_completed(futures):
            pdf = futures[fut]
            processed += 1
            try:
                chunks = fut.result()
                all_chunks.extend(chunks)
            except Exception as e:
                log.error(f"FAIL {os.path.basename(pdf)}: {e}")
                failed += 1
                failed_paths.append(pdf)

            if processed % PROGRESS_EVERY == 0:
                elapsed = time.time() - start
                rate = processed / elapsed
                eta = (len(pdfs) - processed) / rate if rate else 0
                log.info(
                    f"Progress: {processed}/{len(pdfs)} PDFs "
                    f"({rate:.1f}/s, ETA {eta/60:.1f}m) — "
                    f"{len(all_chunks):,} chunks, {failed} failed"
                )

    elapsed = time.time() - start
    log.info(f"Extraction done in {elapsed/60:.1f}m")
    log.info(f"  PDFs: {processed - failed} ok / {failed} failed")
    log.info(f"  Chunks: {len(all_chunks):,}")

    out_chunks = os.path.join(OUTPUT_DIR, "all_chunks.json")
    log.info(f"Writing {out_chunks} ...")
    with open(out_chunks, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    log.info(f"  Size: {os.path.getsize(out_chunks)/1024/1024:.1f} MB")

    manifest = {
        "pdfs_listed": len(pdfs),
        "pdfs_processed": processed - failed,
        "pdfs_failed": failed,
        "total_chunks": len(all_chunks),
        "extraction_seconds": int(elapsed),
        "pdf_root": PDF_ROOT,
        "failed_pdfs": [os.path.basename(p) for p in failed_paths],
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("=" * 60)
    log.info(f"DONE — {len(all_chunks):,} chunks in {elapsed/60:.1f}m")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
