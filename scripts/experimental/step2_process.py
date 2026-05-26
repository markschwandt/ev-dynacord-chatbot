"""
STEP 2: Process PDFs one at a time from the file list. Ultra low memory.
Writes chunks to disk as it goes. Then builds the vector store.
Usage: python3 scripts/step2_process.py
"""
import os, sys, json, hashlib, gc, pickle
from pathlib import Path
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pdf_list = os.path.join(project_dir, "data", "pdf_list.txt")
chunks_jsonl = os.path.join(project_dir, "data", "chunks_temp.jsonl")
chunks_dir = os.path.join(project_dir, "data", "chunks")
vs_dir = os.path.join(project_dir, "data", "vectorstore")
os.makedirs(chunks_dir, exist_ok=True)
os.makedirs(vs_dir, exist_ok=True)

CHUNK_SIZE = 1000
OVERLAP = 200
MIN_SZ = 50

if not os.path.exists(pdf_list):
    print("❌ Run step1_list_pdfs.py first!")
    sys.exit(1)

with open(pdf_list) as f:
    pdfs = [l.strip() for l in f if l.strip()]

# Detect root (common parent of all PDFs)
root = os.path.commonpath(pdfs) if pdfs else "."

print(f"📄 Processing {len(pdfs)} PDFs one at a time...\n")

success = 0
failed = 0
total_chunks = 0

with open(chunks_jsonl, "w", encoding="utf-8") as out:
    for i, fp in enumerate(pdfs):
        try:
            doc = fitz.open(fp)
            text = ""
            for page in doc:
                t = page.get_text("text")
                if t.strip():
                    text += t + "\n\n"
            doc.close()
            del doc
        except Exception:
            failed += 1
            if (i + 1) % 25 == 0:
                gc.collect()
            continue

        if len(text.strip()) < MIN_SZ:
            failed += 1
            del text
            continue

        # Derive metadata
        rel = os.path.relpath(fp, root)
        parts = Path(rel).parts
        brand = "Dynacord" if "Dynacord" in rel else "Electro-Voice"
        cat = parts[1] if len(parts) > 2 else parts[0] if len(parts) > 1 else "General"
        meta = {"brand": brand, "category": cat, "filename": parts[-1], "relative_path": rel}

        # Chunk
        text = text.strip()
        start = 0
        j = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            if end >= len(text):
                c = text[start:]
                if len(c.strip()) >= MIN_SZ:
                    cid = hashlib.md5(f"{rel}:{j}".encode()).hexdigest()
                    out.write(json.dumps({"id": cid, "text": c.strip(), "metadata": {**meta, "chunk_index": j}}, ensure_ascii=False) + "\n")
                    total_chunks += 1
                break
            b = text.rfind(". ", start, end)
            if b <= start: b = text.rfind(" ", start, end)
            if b <= start: b = end
            c = text[start:b+1].strip()
            if len(c) >= MIN_SZ:
                cid = hashlib.md5(f"{rel}:{j}".encode()).hexdigest()
                out.write(json.dumps({"id": cid, "text": c, "metadata": {**meta, "chunk_index": j}}, ensure_ascii=False) + "\n")
                total_chunks += 1
                j += 1
            start = b + 1 - OVERLAP
            if start <= 0: start = b + 1

        del text
        success += 1

        if (i + 1) % 25 == 0:
            gc.collect()
        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(pdfs)} | {success} ok | {total_chunks} chunks")

print(f"\n   ✅ {success} PDFs → {total_chunks} chunks (skipped {failed})")

# ── Build TF-IDF ──────────────────────────────────────────────────────────
print("\n🔢 Building vector store...")

texts, metadatas, ids = [], [], []
with open(chunks_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        texts.append(r["text"])
        metadatas.append(r["metadata"])
        ids.append(r["id"])

print(f"   {len(texts)} chunks loaded")

vectorizer = TfidfVectorizer(
    max_features=50000, stop_words="english",
    ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95,
)
tfidf_matrix = vectorizer.fit_transform(texts)

with open(os.path.join(vs_dir, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)
with open(os.path.join(vs_dir, "tfidf_matrix.pkl"), "wb") as f:
    pickle.dump(tfidf_matrix, f)
with open(os.path.join(vs_dir, "chunks_meta.json"), "w", encoding="utf-8") as f:
    json.dump({"texts": texts, "metadatas": metadatas, "ids": ids}, f, ensure_ascii=False)

# Save standard chunks file too
with open(chunks_jsonl) as f:
    all_c = [json.loads(l) for l in f]
with open(os.path.join(chunks_dir, "all_chunks.json"), "w", encoding="utf-8") as f:
    json.dump(all_c, f, ensure_ascii=False)

os.remove(chunks_jsonl)

print(f"   ✅ {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features")
print(f"\n{'='*60}")
print(f"🎉 DONE! {success}/{len(pdfs)} PDFs → {total_chunks} chunks")
print(f"\n🚀 Run the chatbot:")
print(f'   export OPENAI_API_KEY="sk-your-key"')
print(f"   streamlit run app/chatbot.py")
print(f"{'='*60}")
