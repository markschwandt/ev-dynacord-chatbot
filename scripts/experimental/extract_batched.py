"""Batch PDF extraction - processes in small batches to avoid OOM."""
import os, json, hashlib, gc, sys
from pathlib import Path
import fitz

root = "/sessions/charming-blissful-turing/mnt/Electro-Voice_Dynacord Website Document Downloads"
outdir = "/sessions/charming-blissful-turing/ev-dynacord-chatbot/data/chunks"
os.makedirs(outdir, exist_ok=True)

CHUNK_SIZE, OVERLAP, MIN_SZ = 1000, 200, 50

def chunk(text):
    chunks = []
    if not text or len(text.strip()) < MIN_SZ: return chunks
    text = text.strip(); start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end >= len(text):
            c = text[start:]
            if len(c.strip()) >= MIN_SZ: chunks.append(c.strip())
            break
        b = text.rfind(". ", start, end)
        if b <= start: b = text.rfind(" ", start, end)
        if b <= start: b = end
        c = text[start:b+1].strip()
        if len(c) >= MIN_SZ: chunks.append(c)
        start = b + 1 - OVERLAP
        if start <= 0: start = b + 1
    return chunks

# Find readable PDFs
pdfs = []
for dp, _, fns in os.walk(root):
    for f in sorted(fns):
        if f.lower().endswith(".pdf"):
            fp = os.path.join(dp, f)
            try:
                with open(fp, 'rb') as fh: fh.read(10)
                pdfs.append(fp)
            except: pass

print(f"Found {len(pdfs)} readable PDFs", flush=True)

all_chunks = []
ok = 0

for i, fp in enumerate(pdfs):
    try:
        doc = fitz.open(fp)
        pages = []
        for p in doc:
            t = p.get_text("text")
            if t.strip(): pages.append(t)
        doc.close()
        del doc
        text = "\n\n".join(pages)
        del pages
        
        if not text: 
            del text
            gc.collect()
            continue

        rel = os.path.relpath(fp, root)
        parts = Path(rel).parts
        brand = "Dynacord" if "Dynacord" in rel else "Electro-Voice"
        cat = parts[1] if len(parts) > 2 else parts[0] if len(parts) > 1 else "General"
        meta = {"brand": brand, "category": cat, "filename": parts[-1], "relative_path": rel}

        for j, c in enumerate(chunk(text)):
            cid = hashlib.md5(f"{rel}:{j}".encode()).hexdigest()
            all_chunks.append({"id": cid, "text": c, "metadata": {**meta, "chunk_index": j}})
        ok += 1
        del text
    except Exception as e:
        pass
    
    # Aggressive GC every 50 files
    if (i+1) % 50 == 0:
        gc.collect()
        print(f"{i+1}/{len(pdfs)} | {ok} ok | {len(all_chunks)} chunks", flush=True)

# Save
out = os.path.join(outdir, "all_chunks.json")
with open(out, "w") as f:
    json.dump(all_chunks, f, ensure_ascii=False)

print(f"DONE: {ok} PDFs -> {len(all_chunks)} chunks -> {out}", flush=True)
