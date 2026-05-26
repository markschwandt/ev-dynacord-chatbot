"""
ONE-STEP INGESTION SCRIPT (Memory-Safe)
Processes PDFs in small batches, saving incrementally to avoid OOM kills.

Usage:
    python3 scripts/ingest.py /path/to/your/pdf/folder

Example:
    python3 scripts/ingest.py "../"
"""

import os
import sys
import json
import hashlib
import pickle
import gc
from pathlib import Path

import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 50
BATCH_SIZE = 25  # Small batches to keep memory low


def find_pdfs(root):
    pdfs = []
    for dp, _, fns in os.walk(root):
        for f in sorted(fns):
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(dp, f))
    return pdfs


def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        del doc
        return "\n\n".join(pages)
    except Exception:
        return ""


def chunk_text(text):
    chunks = []
    if not text or len(text.strip()) < MIN_CHUNK_SIZE:
        return chunks
    text = text.strip()
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end >= len(text):
            c = text[start:]
            if len(c.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(c.strip())
            break
        b = text.rfind(". ", start, end)
        if b <= start:
            b = text.rfind(" ", start, end)
        if b <= start:
            b = end
        c = text[start:b + 1].strip()
        if len(c) >= MIN_CHUNK_SIZE:
            chunks.append(c)
        start = b + 1 - CHUNK_OVERLAP
        if start <= 0:
            start = b + 1
    return chunks


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
    if len(sys.argv) < 2:
        print('Usage: python3 scripts/ingest.py "/path/to/pdf/folder"')
        sys.exit(1)

    pdf_root = os.path.abspath(sys.argv[1])
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks_dir = os.path.join(project_dir, "data", "chunks")
    vectorstore_dir = os.path.join(project_dir, "data", "vectorstore")
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(vectorstore_dir, exist_ok=True)

    chunks_file = os.path.join(chunks_dir, "all_chunks.json")

    # ── Step 1: Find PDFs ──────────────────────────────────────────────────
    print(f"\n📁 Scanning: {pdf_root}")
    pdfs = find_pdfs(pdf_root)
    print(f"   Found {len(pdfs)} PDF files\n")

    if not pdfs:
        print("❌ No PDF files found. Check the path and try again.")
        sys.exit(1)

    # ── Step 2: Extract & Chunk in batches ─────────────────────────────────
    # We write chunks to a file incrementally using JSON lines,
    # then read them back for vectorization.
    print("📄 Extracting text from PDFs (in batches of 25)...")

    temp_chunks_file = os.path.join(chunks_dir, "chunks_temp.jsonl")
    total_chunks = 0
    success = 0
    failed = 0

    with open(temp_chunks_file, "w", encoding="utf-8") as out:
        for i, fp in enumerate(pdfs):
            text = extract_text(fp)
            if not text:
                failed += 1
                del text
                # Force GC every batch
                if (i + 1) % BATCH_SIZE == 0:
                    gc.collect()
                continue

            metadata = derive_metadata(fp, pdf_root)
            chunks = chunk_text(text)
            del text  # Free memory immediately

            if not chunks:
                failed += 1
                continue

            for j, c in enumerate(chunks):
                cid = hashlib.md5(f"{metadata['relative_path']}:{j}".encode()).hexdigest()
                record = {
                    "id": cid,
                    "text": c,
                    "metadata": {**metadata, "chunk_index": j},
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

            del chunks
            success += 1

            if (i + 1) % BATCH_SIZE == 0:
                gc.collect()

            if (i + 1) % 200 == 0:
                print(f"   {i+1}/{len(pdfs)} processed | {success} with text | {total_chunks} chunks")

    print(f"   ✅ {success} PDFs → {total_chunks} chunks (skipped {failed})\n")

    # ── Step 3: Read chunks back and build vector store ────────────────────
    print("🔢 Building TF-IDF vector store...")
    print("   Loading chunks from disk...")

    texts = []
    metadatas = []
    ids = []

    with open(temp_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
            metadatas.append(record["metadata"])
            ids.append(record["id"])

    print(f"   Loaded {len(texts)} chunks into memory")
    print("   Fitting TF-IDF model...")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    print(f"   Saving vector store...")

    with open(os.path.join(vectorstore_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(vectorstore_dir, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(os.path.join(vectorstore_dir, "chunks_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metadatas": metadatas, "ids": ids}, f, ensure_ascii=False)

    # Also save the standard all_chunks.json
    all_chunks = []
    with open(temp_chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            all_chunks.append(json.loads(line))
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    del all_chunks

    # Clean up temp file
    os.remove(temp_chunks_file)

    print(f"   ✅ Vector store: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features\n")

    # ── Done ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"🎉 INGESTION COMPLETE!")
    print(f"   PDFs processed:  {success}/{len(pdfs)}")
    print(f"   Total chunks:    {total_chunks}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"\n🚀 Now run:")
    print(f'   export OPENAI_API_KEY="sk-your-key"')
    print(f"   streamlit run app/chatbot.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
