"""
Upload all EV/Dynacord PDFs from Drive sync to a Gemini File Search Store.

- Reads existing store name from data/gemini_store_name.txt (created by smoke test).
- Walks Drive sync path for all .pdf files.
- Skips PDFs already in the store (filename match — resume-safe).
- Uploads with bounded concurrency (8 in flight), polls each operation to completion.
- Logs progress + failures.
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from google import genai
from google.genai.errors import APIError

PDF_ROOT = "/Users/vivaknievel/Library/CloudStorage/GoogleDrive-schwandt.mark@gmail.com/My Drive/Electro-Voice_Dynacord Website Document Downloads"
STORE_NAME_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'gemini_store_name.txt')
WORKERS = 8
POLL_INTERVAL = 3
POLL_TIMEOUT = 300  # 5 min per file

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

with open(STORE_NAME_FILE) as f:
    STORE = f.read().strip()
print(f"Store: {STORE}", flush=True)

# Index already-uploaded filenames so we can resume
print("Listing existing documents in store...", flush=True)
existing = set()
for d in client.file_search_stores.documents.list(parent=STORE):
    if d.display_name:
        existing.add(d.display_name)
print(f"  {len(existing)} already in store", flush=True)

# Walk Drive for all PDFs
print("Walking Drive for PDFs...", flush=True)
all_pdfs = []
for dirpath, _, files in os.walk(PDF_ROOT):
    for f in files:
        if f.lower().endswith('.pdf'):
            all_pdfs.append(os.path.join(dirpath, f))
print(f"  Found {len(all_pdfs)} PDFs in Drive", flush=True)

to_upload = [p for p in all_pdfs if os.path.basename(p) not in existing]
print(f"  {len(to_upload)} PDFs need uploading ({len(all_pdfs) - len(to_upload)} skipped)", flush=True)

if not to_upload:
    print("Nothing to do.", flush=True)
    sys.exit(0)

# Upload with concurrency
counter_lock = threading.Lock()
done = 0
failed = 0
start = time.time()


def upload_one(pdf_path):
    global done, failed
    name = os.path.basename(pdf_path)
    try:
        op = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=STORE,
            file=pdf_path,
        )
        deadline = time.time() + POLL_TIMEOUT
        while not op.done and time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            op = client.operations.get(op)
        if not op.done:
            raise TimeoutError(f"timed out after {POLL_TIMEOUT}s")
        result = 'ok'
    except Exception as e:
        result = f"FAIL: {type(e).__name__}: {str(e)[:200]}"

    with counter_lock:
        global done, failed
        done += 1
        if result != 'ok':
            failed += 1
        if done % 25 == 0 or result != 'ok':
            elapsed = time.time() - start
            rate = done / elapsed if elapsed else 0
            eta = (len(to_upload) - done) / rate if rate else 0
            print(f"  [{done}/{len(to_upload)}] {name}: {result}  "
                  f"({rate:.1f}/s, ETA {eta/60:.1f}m, {failed} failed)", flush=True)


print(f"\nUploading with {WORKERS} concurrent workers...\n", flush=True)
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = [ex.submit(upload_one, p) for p in to_upload]
    for f in as_completed(futures):
        pass  # results logged inside upload_one

elapsed = time.time() - start
print(f"\nDONE — {done - failed}/{len(to_upload)} uploaded ({failed} failed) in {elapsed/60:.1f}m", flush=True)
