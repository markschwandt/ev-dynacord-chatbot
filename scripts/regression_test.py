"""
Regression test for the TF-IDF vectorstore.

Tests the 4 queries that failed in the Mar 17 2026 compliance report:
  - EVERSE 8 (PDF unprocessed → returned "no documentation")
  - EKX-15P (PDF unprocessed)
  - RE20 (PDF unprocessed)
  - Evolve 50 (TF-IDF noise — "evolve" too common)

For each query, retrieves top-K results and PASSES if the canonical
Engineering Data Sheet / Manual for that product appears in the top 3.

Usage:
    python3 scripts/regression_test.py
"""

import os
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VS_DIR = os.path.join(PROJECT_DIR, "data", "vectorstore")

# product → list of substrings that, if present in a top-3 result's filename
# (case-insensitive), counts as a PASS
TESTS = [
    {
        "query": "What are the specs for the EVERSE 8?",
        "must_match": ["everse 8", "everse-8"],
        "preferred": ["engineering data sheet", "data sheet", "datasheet", "manual"],
    },
    {
        "query": "Show me the engineering data sheet for the EKX-15P.",
        "must_match": ["ekx-15p", "ekx 15p"],
        "preferred": ["engineering data sheet", "data sheet", "datasheet"],
    },
    {
        "query": "Tell me about the EV RE20 microphone — frequency response, polar pattern.",
        "must_match": ["re20", "re-20"],
        "preferred": ["engineering data sheet", "data sheet", "datasheet", "service data"],
    },
    {
        "query": "What are the complete specs for the Evolve 50?",
        "must_match": ["evolve 50", "evolve-50", "evolve50"],
        "preferred": ["engineering data sheet", "data sheet", "datasheet", "manual"],
    },
]


def load_store():
    with open(os.path.join(VS_DIR, "vectorizer.pkl"), "rb") as f:
        vec = pickle.load(f)
    with open(os.path.join(VS_DIR, "tfidf_matrix.pkl"), "rb") as f:
        mat = pickle.load(f)
    with open(os.path.join(VS_DIR, "chunks_meta.json")) as f:
        d = json.load(f)
    return vec, mat, d["metadatas"]


def top_k_filenames(vec, mat, metas, query, k=5):
    q = vec.transform([query])
    sims = cosine_similarity(q, mat).flatten()
    top = sims.argsort()[-k:][::-1]
    return [(sims[i], metas[i].get("filename", "?")) for i in top]


def evaluate(test, top_n):
    """top_n is the list of (score, filename) tuples we retrieved."""
    top_3 = top_n[:3]
    must = test["must_match"]
    pref = test["preferred"]

    has_product = any(
        any(m.lower() in fn.lower() for m in must) for _, fn in top_3
    )
    has_preferred_doc = any(
        any(p.lower() in fn.lower() for p in pref)
        and any(m.lower() in fn.lower() for m in must)
        for _, fn in top_3
    )

    if has_preferred_doc:
        return "PASS"
    elif has_product:
        return "PARTIAL (product in top-3 but not the preferred doc type)"
    else:
        return "FAIL (product not in top-3)"


def main():
    print(f"\n{'='*72}")
    print(f"Regression test — vectorstore at {VS_DIR}")
    print(f"{'='*72}\n")
    vec, mat, metas = load_store()
    print(f"Index: {mat.shape[0]:,} chunks × {mat.shape[1]:,} features\n")

    passed = 0
    for t in TESTS:
        top = top_k_filenames(vec, mat, metas, t["query"], k=5)
        verdict = evaluate(t, top)
        print(f"Q: {t['query']}")
        print(f"   → {verdict}")
        for score, fn in top:
            print(f"      {score:.4f}  {fn}")
        print()
        if verdict == "PASS":
            passed += 1

    print(f"{'='*72}")
    print(f"RESULT: {passed}/{len(TESTS)} queries PASS")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
