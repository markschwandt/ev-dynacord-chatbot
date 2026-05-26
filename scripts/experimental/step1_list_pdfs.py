"""
STEP 1: Just find and list all PDF file paths. Lightweight — no PDF processing.
Usage: python3 scripts/step1_list_pdfs.py "/path/to/pdfs"
"""
import os, sys

if len(sys.argv) < 2:
    print('Usage: python3 scripts/step1_list_pdfs.py "/path/to/pdfs"')
    sys.exit(1)

root = os.path.abspath(sys.argv[1])
outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pdf_list.txt")
os.makedirs(os.path.dirname(outfile), exist_ok=True)

print(f"Scanning: {root}")
count = 0
with open(outfile, "w") as f:
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if fn.lower().endswith(".pdf"):
                f.write(os.path.join(dp, fn) + "\n")
                count += 1
                if count % 500 == 0:
                    print(f"  Found {count} so far...")

print(f"\n✅ Found {count} PDFs. List saved to data/pdf_list.txt")
print(f"Now run: python3 scripts/step2_process.py")
