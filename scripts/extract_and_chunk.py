"""
PDF Text Extraction and Chunking Script
Extracts text from all PDFs in the Electro-Voice/Dynacord document library,
chunks them into manageable pieces, and saves as JSON for embedding.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Generator

import fitz  # PyMuPDF - fast and reliable PDF extraction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("extraction.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
CHUNK_SIZE = 1000       # Target chunk size in characters
CHUNK_OVERLAP = 200     # Overlap between chunks for context continuity
MIN_CHUNK_SIZE = 50     # Minimum chunk size to keep (filters noise)
PDF_ROOT = os.environ.get(
    "PDF_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "data", "pdfs"),
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "chunks"),
)
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path}: {e}")
        return ""


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Generator[str, None, None]:
    """
    Split text into overlapping chunks, trying to break on paragraph
    or sentence boundaries for cleaner context.
    """
    if not text or len(text.strip()) < MIN_CHUNK_SIZE:
        return

    # Normalize whitespace
    text = text.strip()

    start = 0
    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:]
            if len(chunk.strip()) >= MIN_CHUNK_SIZE:
                yield chunk.strip()
            break

        # Try to break at paragraph boundary
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start:
            # Try sentence boundary
            boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            # Try any newline
            boundary = text.rfind("\n", start, end)
        if boundary == -1 or boundary <= start:
            # Try any space
            boundary = text.rfind(" ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end

        chunk = text[start : boundary + 1].strip()
        if len(chunk) >= MIN_CHUNK_SIZE:
            yield chunk

        start = boundary + 1 - overlap
        if start <= 0:
            start = boundary + 1


def derive_metadata(pdf_path: str, root: str) -> dict:
    """
    Extract useful metadata from the file path.
    E.g., brand (EV vs Dynacord), category, subcategory, filename.
    """
    rel = os.path.relpath(pdf_path, root)
    parts = Path(rel).parts

    brand = "Unknown"
    if "Dynacord" in rel:
        brand = "Dynacord"
    elif "Electro-Voice" in rel or parts[0].startswith("Marketing") or parts[0].startswith("Technical") or parts[0].startswith("Software"):
        brand = "Electro-Voice"

    category = parts[1] if len(parts) > 2 else parts[0] if len(parts) > 1 else "General"
    subcategory = parts[2] if len(parts) > 3 else ""
    filename = parts[-1]

    return {
        "brand": brand,
        "category": category,
        "subcategory": subcategory,
        "filename": filename,
        "relative_path": rel,
    }


def process_single_pdf(args: tuple) -> list[dict]:
    """Process a single PDF: extract text, chunk, and return chunk dicts."""
    pdf_path, root = args
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return []

    metadata = derive_metadata(pdf_path, root)
    chunks = []

    for i, chunk_text_str in enumerate(chunk_text(text)):
        chunk_id = hashlib.md5(
            f"{metadata['relative_path']}:{i}".encode()
        ).hexdigest()

        chunks.append({
            "id": chunk_id,
            "text": chunk_text_str,
            "metadata": {
                **metadata,
                "chunk_index": i,
                "char_count": len(chunk_text_str),
            },
        })

    return chunks


def find_all_pdfs(root: str) -> list[str]:
    """Recursively find all PDF files under root."""
    pdfs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dirpath, f))
    return sorted(pdfs)


def main():
    pdf_root = os.path.abspath(PDF_ROOT)
    output_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Scanning for PDFs in: {pdf_root}")
    pdf_files = find_all_pdfs(pdf_root)
    logger.info(f"Found {len(pdf_files)} PDF files")

    if not pdf_files:
        logger.error("No PDF files found. Check PDF_ROOT path.")
        return

    all_chunks = []
    failed = 0
    processed = 0

    logger.info(f"Processing with {MAX_WORKERS} workers...")

    args_list = [(pdf, pdf_root) for pdf in pdf_files]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_pdf, args): args[0]
            for args in args_list
        }

        for future in as_completed(futures):
            pdf_path = futures[future]
            processed += 1

            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                failed += 1

            if processed % 100 == 0:
                logger.info(
                    f"Progress: {processed}/{len(pdf_files)} PDFs processed, "
                    f"{len(all_chunks)} chunks so far"
                )

    # Save all chunks to a single JSON file
    output_path = os.path.join(output_dir, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Save a manifest/summary
    manifest = {
        "total_pdfs_found": len(pdf_files),
        "total_pdfs_processed": processed - failed,
        "total_pdfs_failed": failed,
        "total_chunks": len(all_chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "output_file": output_path,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Extraction complete!")
    logger.info(f"  PDFs processed: {processed - failed}/{len(pdf_files)}")
    logger.info(f"  PDFs failed:    {failed}")
    logger.info(f"  Total chunks:   {len(all_chunks)}")
    logger.info(f"  Output:         {output_path}")
    logger.info(f"  Manifest:       {manifest_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
