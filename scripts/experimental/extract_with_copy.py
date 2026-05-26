"""
PDF Text Extraction - copies each PDF to /tmp before processing
to work around mounted filesystem locking issues.
"""

import os
import json
import hashlib
import shutil
import tempfile
import logging
from pathlib import Path

import fitz

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
    """Copy PDF to temp dir, extract text, clean up."""
    tmp_path = None
    try:
        # Copy to temp to avoid mount filesystem locks
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        shutil.copy2(pdf_path, tmp_path)

        doc = fitz.open(tmp_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


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
    brand = "Dynacord" if "Dynacord" in rel else "Electro-Voice"
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

    pdfs = []
    for dirpath, _, filenames in os.walk(pdf_root):
        for f in filenames:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dirpath, f))
    pdfs.sort()
    logger.info(f"Found {len(pdfs)} PDFs")

    all_chunks = []
    success = 0
    failed = 0
    no_text = 0

    for i, pdf_path in enumerate(pdfs):
        text = extract_text(pdf_path)
        if not text:
            failed += 1
            continue

        metadata = derive_metadata(pdf_path, pdf_root)
        chunk_count = 0
        for j, chunk in enumerate(chunk_text(text)):
            chunk_id = hashlib.md5(f"{metadata['relative_path']}:{j}".encode()).hexdigest()
            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {**metadata, "chunk_index": j, "char_count": len(chunk)},
            })
            chunk_count += 1

        if chunk_count == 0:
            no_text += 1
        else:
            success += 1

        if (i + 1) % 200 == 0:
            logger.info(
                f"Progress: {i+1}/{len(pdfs)} | "
                f"Success: {success} | Failed: {failed} | "
                f"Chunks: {len(all_chunks)}"
            )

    # Save chunks
    output_path = os.path.join(output_dir, "all_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Done! {success} PDFs with text → {len(all_chunks)} chunks")
    logger.info(f"Failed to open: {failed} | No usable text: {no_text}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
