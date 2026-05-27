"""
Upload all EV/Dynacord PDFs from Drive sync to a Gemini File Search Store.

- Reads existing store name from data/gemini_store_name.txt (created by smoke test).
- Walks Drive sync path for all .pdf files.
- Skips PDFs already in the store (filename match — resume-safe; matches against
  BOTH the original basename and the ASCII-sanitized basename, so reruns after the
  Unicode patch don't double-upload).
- Uploads with bounded concurrency (8 in flight), polls each operation to completion.
- Non-ASCII filenames (bullet `•`, non-breaking hyphen `‑`, etc.) are uploaded via a
  symlink in a tempdir with a sanitized name — the google-genai SDK ASCII-encodes
  filenames in HTTP headers and crashes otherwise.
- Transient 500 INTERNAL responses are retried up to 3 times with exponential backoff.
"""

import os
import re
import sys
import time
import shutil
import tempfile
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from google import genai
from google.genai.errors import APIError, ServerError

PDF_ROOT = "/Users/vivaknievel/Library/CloudStorage/GoogleDrive-schwandt.mark@gmail.com/My Drive/Electro-Voice_Dynacord Website Document Downloads"
STORE_NAME_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'gemini_store_name.txt')
WORKERS = 8
POLL_INTERVAL = 3
POLL_TIMEOUT = 300  # 5 min per file
MAX_RETRIES = 3     # for transient 500s
RETRY_BACKOFF = 5   # seconds, doubled each retry

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

with open(STORE_NAME_FILE) as f:
    STORE = f.read().strip()
print(f"Store: {STORE}", flush=True)


def sanitize_filename(name):
    """Replace non-ASCII chars with closest ASCII equivalent or _.
    NFKD splits e.g. 'é' → 'e' + combining accent; encode('ascii', 'ignore') drops the accent.
    Any remaining non-ASCII (like •, ‑) is replaced with _.
    """
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_bytes = nfkd.encode('ascii', 'ignore').decode('ascii')
    # If sanitization dropped chars, replace originals with _ for readability
    if ascii_bytes != name:
        # rebuild char-by-char: keep ASCII, replace non-ASCII with _
        out = []
        for ch in name:
            try:
                ch.encode('ascii')
                out.append(ch)
            except UnicodeEncodeError:
                out.append('_')
        return ''.join(out)
    return name


# Index already-uploaded filenames so we can resume.
# Check both raw and sanitized names because earlier runs may have stored
# under the sanitized name after the patch.
print("Listing existing documents in store...", flush=True)
existing = set()
for d in client.file_search_stores.documents.list(parent=STORE):
    if d.display_name:
        existing.add(d.display_name)
        existing.add(sanitize_filename(d.display_name))
print(f"  {len(existing)} display-name variants already in store", flush=True)

# Walk Drive for non-LEGACY Engineering Data Sheets only.
# Scope decision (2026-05-26): BROAD EDS-only — keeps current/legacy product EDSs,
# excludes manuals/certs/brochures/software bundles to fit Gemini Tier 1 10 GB cap.
EDS_PAT = re.compile(r"(?i)engineering data sheet|spec sheet|specification sheet|datasheet")
LEGACY_PAT = re.compile(r"(?i)\b(legacy|end of life|eol|discontinued|superceded|superseded|retired)\b")

print("Walking Drive for non-LEGACY Engineering Data Sheets only...", flush=True)
all_pdfs = []
skipped_non_eds = 0
skipped_legacy = 0
for dirpath, _, files in os.walk(PDF_ROOT):
    for f in files:
        if not f.lower().endswith('.pdf'):
            continue
        if not EDS_PAT.search(f):
            skipped_non_eds += 1
            continue
        if LEGACY_PAT.search(f):
            skipped_legacy += 1
            continue
        all_pdfs.append(os.path.join(dirpath, f))
print(f"  Found {len(all_pdfs)} EDS PDFs (skipped {skipped_non_eds} non-EDS, {skipped_legacy} legacy)", flush=True)

to_upload = []
for p in all_pdfs:
    base = os.path.basename(p)
    if base in existing or sanitize_filename(base) in existing:
        continue
    to_upload.append(p)
print(f"  {len(to_upload)} PDFs need uploading ({len(all_pdfs) - len(to_upload)} skipped)", flush=True)

if not to_upload:
    print("Nothing to do.", flush=True)
    sys.exit(0)

# Tempdir for symlinks with sanitized names (one per worker — avoid collision)
TEMP_DIR = tempfile.mkdtemp(prefix='ev_dc_upload_')
print(f"Tempdir for sanitized symlinks: {TEMP_DIR}", flush=True)

# Upload with concurrency
counter_lock = threading.Lock()
done = 0
failed = 0
start = time.time()


def upload_one(pdf_path):
    global done, failed
    name = os.path.basename(pdf_path)
    sanitized = sanitize_filename(name)
    needs_symlink = (sanitized != name)
    upload_path = pdf_path
    tmp_link = None

    if needs_symlink:
        # Make a tempfile-named symlink with the sanitized name
        tmp_link = os.path.join(TEMP_DIR, f"{os.getpid()}_{threading.get_ident()}_{sanitized}")
        try:
            if os.path.exists(tmp_link):
                os.unlink(tmp_link)
            os.symlink(pdf_path, tmp_link)
            upload_path = tmp_link
        except OSError as e:
            # Fall back to copy if symlink fails
            shutil.copy2(pdf_path, tmp_link)
            upload_path = tmp_link

    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            op = client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=STORE,
                file=upload_path,
            )
            deadline = time.time() + POLL_TIMEOUT
            while not op.done and time.time() < deadline:
                time.sleep(POLL_INTERVAL)
                op = client.operations.get(op)
            if not op.done:
                raise TimeoutError(f"timed out after {POLL_TIMEOUT}s")
            result = 'ok' + (' (sanitized)' if needs_symlink else '')
            break
        except ServerError as e:
            # 500/503 — retryable
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * (2 ** (attempt - 1)))
                continue
            result = f"FAIL after {MAX_RETRIES} retries: ServerError: {str(e)[:150]}"
        except Exception as e:
            result = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
            break

    # Cleanup symlink
    if tmp_link and os.path.exists(tmp_link):
        try:
            os.unlink(tmp_link)
        except OSError:
            pass

    with counter_lock:
        global done, failed
        done += 1
        if not result.startswith('ok'):
            failed += 1
        if done % 25 == 0 or not result.startswith('ok'):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed else 0
            eta = (len(to_upload) - done) / rate if rate else 0
            print(f"  [{done}/{len(to_upload)}] {name}: {result}  "
                  f"({rate:.1f}/s, ETA {eta/60:.1f}m, {failed} failed)", flush=True)


print(f"\nUploading with {WORKERS} concurrent workers (sanitized-name fix + 3x retry on 500)...\n", flush=True)
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = [ex.submit(upload_one, p) for p in to_upload]
    for f in as_completed(futures):
        pass  # results logged inside upload_one

elapsed = time.time() - start
print(f"\nDONE — {done - failed}/{len(to_upload)} uploaded ({failed} failed) in {elapsed/60:.1f}m", flush=True)

# Cleanup tempdir
try:
    shutil.rmtree(TEMP_DIR)
except OSError:
    pass
