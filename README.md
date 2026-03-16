# Electro-Voice / Dynacord Document Search & Chatbot

A searchable knowledge base built from 3,080 Electro-Voice and Dynacord product PDFs.

## Quick Start

### HTML Search (no installation needed)
Download and open `ev-dynacord-search.html` in Chrome. Search across 63,167 document chunks instantly.

### Chatbot (requires Python + OpenAI API key)
```bash
pip install streamlit openai scikit-learn
cd app
streamlit run chatbot.py
```

## Data

- **63,167** text chunks from **3,080** PDFs
- TF-IDF vector store with 50,000 features
- Brands: Electro-Voice, Dynacord
- Categories: Technical Documentation, Marketing Materials, Software/Firmware

## Structure

```
data/
  chunks/all_chunks.json        # Raw text chunks with metadata
  vectorstore/                   # TF-IDF vectors for search
    vectorizer.pkl               # Fitted TF-IDF vectorizer
    tfidf_matrix.pkl             # Sparse TF-IDF matrix
    chunks_meta.json             # Texts + metadata for retrieval
app/
  chatbot.py                     # Streamlit chatbot with OpenAI
ev-dynacord-search.html          # Self-contained search app
```

## How It Works

1. Text was extracted from all PDFs using PyMuPDF
2. Text was split into ~1000 character chunks with 200 character overlap
3. TF-IDF vectors (50K vocabulary, bigrams, sublinear TF) were built with scikit-learn
4. The chatbot uses cosine similarity to find relevant chunks, then sends them to OpenAI for answers
5. The HTML search app embeds all chunks as gzip-compressed data for client-side search
