"""
TF-IDF Vector Store Builder.

Reads data/chunks/all_chunks.json, builds a TF-IDF index with the same
hyperparameters the original Mar 2026 deployment uses, and writes:
  data/vectorstore/vectorizer.pkl    (sklearn TfidfVectorizer)
  data/vectorstore/tfidf_matrix.pkl  (sparse matrix)
  data/vectorstore/chunks_meta.json  ({texts, metadatas, ids})

These are exactly what app/chatbot.py's `load_vectorstore()` expects.

Usage:
    python3 scripts/build_tfidf.py
"""

import os
import json
import pickle
import time
import logging

from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_FILE = os.path.join(PROJECT_DIR, "data", "chunks", "all_chunks.json")
VS_DIR = os.path.join(PROJECT_DIR, "data", "vectorstore")

# Same hyperparameters as step2_process.py (the script that built the deployed index)
VECTORIZER_KWARGS = dict(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main():
    os.makedirs(VS_DIR, exist_ok=True)

    log.info(f"Loading chunks from {CHUNKS_FILE}")
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)
    log.info(f"  {len(chunks):,} chunks loaded")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]

    log.info(f"Fitting TfidfVectorizer({VECTORIZER_KWARGS}) ...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(**VECTORIZER_KWARGS)
    matrix = vectorizer.fit_transform(texts)
    log.info(f"  matrix shape={matrix.shape}, nnz={matrix.nnz:,}, "
             f"vocab={len(vectorizer.vocabulary_):,} terms "
             f"({time.time()-t0:.1f}s)")

    log.info("Writing artifacts ...")
    with open(os.path.join(VS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(VS_DIR, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(matrix, f)
    with open(os.path.join(VS_DIR, "chunks_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metadatas": metadatas, "ids": ids}, f, ensure_ascii=False)

    for name in ("vectorizer.pkl", "tfidf_matrix.pkl", "chunks_meta.json"):
        sz = os.path.getsize(os.path.join(VS_DIR, name)) / 1024 / 1024
        log.info(f"  {name}: {sz:.1f} MB")

    log.info("=" * 60)
    log.info(f"DONE — {matrix.shape[0]:,} chunks × {matrix.shape[1]:,} features")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
