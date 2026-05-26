"""
PDF Text Extraction - Single-threaded version for mounted filesystems.
"""

import os
import json
import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 50

PDF_ROOT = os.environ.get("PDF_ROOT", "./data/pdfs")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./data/chunks")


def extract_text(pdf_path):
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
        return ""


def chunk_text(text):
    if not text or len(text.strip()) < MIN_CHUNK_SIZE:
        return
    text = text.strip()
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end >= len(text):
            chunk = text[start:]
            if len(chunk.strip()) >= MIN_CHUNK_SIZE:
                yield chunk.strip()
            break
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = text.rfind("\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = text.rfind(" ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end
        chunk = text[start:boundary + 1].strip()
        if len(chunk) >= MIN_CHUNK_SIZE:
            yield chunk
        start = boundary + 1 - CHUNK_OVERLAP
        if start <= 0:
            start = boundary + 1


def derive_metadata(pdf_path, root):
    rel = os.path.relpath(pdf_path, root)
    parts = Path(rel).parts
    brand = "Unknown"
    if "Dynacord" in rel:
        brand = "Dynacord"
    else:
        brand = "Electro-Voice"
    category = parts[1] if len(parts) > 2 else parts[0] if len(parts) > 1 else "General"
    subcategory = parts[2] if len(parts) > 3 else ""
    return {
        "brand": brand,
        "category": category,
        "subcategory": subcategory,
        "filename": parts[-1],
        "relative_path": rel,
    }


def main():
    pdf_root = os.path.abspath(PDF_ROOT)
    output_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # Find all PDFs
    pdfs = []
    for dirpath, _, filenames in os.walk(pdf_root):
        for f in filenames:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dirpath, f))
    pdfs.sort()
    logger.info(f"Found {len(pdfs)} PDFs in {pdf_root}")

    all_chunks = []
    success = 0
    failed = 0

    for i, pdf_path in enumerate(pdfs):
        text = extract_text(pdf_path)
        if not text:
            failed += 1
            if failed <= 10:
                logger.warning(f"No text from: {os.path.basename(pdf_path)}")
            continue

        metadata = derive_metadata(pdf_path, pdf_root)
        for j, chunk in enumerate(chunk_text(text)):
            chunk_id = hashlib.md5(f"{metadata['relative_path']}:{j}".encode()).hexdigest()
            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {**metadata, "chunk_index": j, "char_count": len(chunk)},
            })

        success += 1
        if (i + 1) % 200 == 0:
            logger.info(f"Progress: {i+1}/{len(pdfs)} | Success: {success} | Chunks: {len(all_chunks)}")

    output_path = os.path.join(output_dir, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Done! {success} PDFs → {len(all_chunks)} chunks")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
