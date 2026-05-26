import os
import json
import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# ── Retrieval mode ────────────────────────────────────────────────────────
# "tfidf" (default, current production) or "chroma" (dense embeddings via OpenAI
# text-embedding-3-small + ChromaDB). Set via env var or Streamlit secrets.
RETRIEVAL_MODE = os.environ.get(
    "RETRIEVAL_MODE",
    st.secrets.get("RETRIEVAL_MODE", "tfidf") if hasattr(st, "secrets") else "tfidf",
)

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vectorstore")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chromadb")
CHROMA_COLLECTION = "ev_dynacord_docs"
EMBEDDING_MODEL = "text-embedding-3-small"

# Stop words to exclude from filename matching — these cause massive noise
STOP_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "had", "her", "was", "one", "our", "out", "has", "his", "how", "its",
    "may", "new", "now", "old", "see", "way", "who", "did", "get", "let",
    "say", "she", "too", "use", "what", "with", "this", "that", "from",
    "they", "been", "have", "many", "some", "them", "than", "each", "make",
    "like", "over", "such", "take", "into", "most", "about", "other",
    "which", "their", "there", "could", "would", "should", "where", "when",
    "tell", "show", "give", "best", "does", "will", "just", "more", "also",
    "very", "much", "what", "specs", "specifications", "information", "details",
    "power", "rating", "frequency", "response", "weight", "dimensions",
}


# ── Doc-type re-ranking (shared between TF-IDF and Chroma modes) ─────────
def _doctype_multiplier(filename: str) -> float:
    """Boost Engineering Data Sheets / manuals, penalize certificates."""
    fn = filename.lower()
    if "engineering data sheet" in fn or "data sheet" in fn:
        return 1.3
    if "user manual" in fn or "owner" in fn:
        return 1.2
    if "spec" in fn or "specification" in fn:
        return 1.2
    if "declaration of conformity" in fn or "certificate" in fn:
        return 0.6
    if "legacy" in fn:
        return 0.7
    return 1.0


# ── TF-IDF mode (current production) ──────────────────────────────────────
@st.cache_resource
def load_tfidf():
    with open(os.path.join(VECTORSTORE_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(VECTORSTORE_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(os.path.join(VECTORSTORE_DIR, "chunks_meta.json"), "r") as f:
        meta = json.load(f)
    return vectorizer, tfidf_matrix, meta


def search_tfidf(query, vectorizer, tfidf_matrix, meta, brand=None, top_k=15):
    """Multi-strategy TF-IDF search with filename matching + doc-type boost."""
    query_lower = query.lower()

    # Strategy 1: Direct TF-IDF search on original query
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Strategy 2: Extract and search product names/model numbers
    product_terms = re.findall(r'[A-Za-z]+[\s\-]?\d+[A-Za-z]*|\b[A-Z]{2,}[- ]?\d+\w*', query)
    if product_terms:
        for term in product_terms:
            term_vec = vectorizer.transform([term])
            term_scores = cosine_similarity(term_vec, tfidf_matrix).flatten()
            scores = np.maximum(scores, term_scores * 0.9)

    # Strategy 3: Filename matching — meaningful query words only
    query_words = [
        w.strip("?.,!") for w in query_lower.split()
        if len(w.strip("?.,!")) > 2 and w.strip("?.,!") not in STOP_WORDS
    ]
    for i, m in enumerate(meta["metadatas"]):
        fname = m.get("filename", "").lower()
        match_count = sum(1 for word in query_words if word in fname)
        if match_count > 0:
            boost = min(0.3 + (match_count - 1) * 0.15, 0.6)
            scores[i] = max(scores[i], boost)

    # Strategy 4: Direct product name match in document text
    if product_terms:
        for i, text in enumerate(meta["texts"]):
            text_upper = text.upper()
            for term in product_terms:
                if term.upper() in text_upper:
                    scores[i] = max(scores[i], 0.25)

    # Brand filter
    if brand:
        for i, m in enumerate(meta["metadatas"]):
            if m.get("brand", "") != brand:
                scores[i] = 0

    # Doc-type prioritization
    for i, m in enumerate(meta["metadatas"]):
        if scores[i] > 0:
            scores[i] *= _doctype_multiplier(m.get("filename", ""))

    top_indices = scores.argsort()[-top_k:][::-1]
    return [
        {"text": meta["texts"][idx], "metadata": meta["metadatas"][idx], "score": float(scores[idx])}
        for idx in top_indices if scores[idx] > 0
    ]


# ── Chroma mode (dense embeddings via OpenAI text-embedding-3-small) ─────
@st.cache_resource
def load_chroma():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(CHROMA_COLLECTION)


@st.cache_resource
def get_embedding_client(api_key):
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def search_chroma(query, collection, api_key, brand=None, top_k=15):
    """Dense-embedding semantic search via ChromaDB."""
    client = get_embedding_client(api_key)
    q_emb = client.embeddings.create(
        input=[query], model=EMBEDDING_MODEL
    ).data[0].embedding

    where = {"brand": brand} if brand else None
    # Over-fetch so doc-type re-ranking has options
    n = min(top_k * 3, 50)
    res = collection.query(
        query_embeddings=[q_emb], n_results=n, where=where,
    )

    if not res["ids"] or not res["ids"][0]:
        return []

    out = []
    for i in range(len(res["ids"][0])):
        # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
        sim = 1.0 - res["distances"][0][i]
        meta = res["metadatas"][0][i]
        score = sim * _doctype_multiplier(meta.get("filename", ""))
        out.append({
            "text": res["documents"][0][i],
            "metadata": meta,
            "score": score,
        })

    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:top_k]


# ── Unified search dispatch ──────────────────────────────────────────────
def search(query, store, api_key=None, brand=None, top_k=15):
    if RETRIEVAL_MODE == "chroma":
        return search_chroma(query, store, api_key, brand=brand, top_k=top_k)
    vectorizer, tfidf_matrix, meta = store
    return search_tfidf(query, vectorizer, tfidf_matrix, meta, brand=brand, top_k=top_k)


def get_unique_sources(results):
    """Get unique document sources from results."""
    seen = set()
    sources = []
    for r in results:
        fname = r["metadata"].get("filename", "Unknown")
        if fname not in seen:
            seen.add(fname)
            sources.append(fname)
    return sources


SYSTEM_PROMPT = """You are the official product support assistant for Electro-Voice (EV) and Dynacord professional audio equipment. You represent these brands exclusively.

CRITICAL RULES - ACCURACY:
1. ONLY state facts that are EXPLICITLY written in the document context below. NEVER invent, guess, or fabricate specifications, features, model numbers, or any other product details. If a specific number (like wattage, frequency response, SPL, weight, etc.) is not explicitly stated in the context documents, DO NOT make one up.
2. If the context documents contain partial information about a product, share ONLY what is actually in the documents and clearly state: "The documents I have access to contain limited information on this product. Here is what I found:" then list only the verified facts.
3. If the context documents do not contain information about the requested product at all, say: "I don't have detailed documentation for that specific product in my current database. Would you like me to help you find information on a similar Electro-Voice or Dynacord product?"

CRITICAL RULES - BRAND EXCLUSIVITY:
4. ONLY recommend and discuss Electro-Voice and Dynacord products. NEVER mention, recommend, or compare with competitor brands (such as Shure, Sennheiser, JBL, QSC, Yamaha, Bose, Rode, Audio-Technica, Allen & Heath, Behringer, Mackie, or any others).
5. If the user asks a general question (like "best microphone for broadcast"), answer ONLY with relevant Electro-Voice or Dynacord products from the document context.
6. If a user asks about a competitor product, politely redirect: "I specialize in Electro-Voice and Dynacord products. I'd be happy to help you find an EV or Dynacord solution for your needs. What application are you looking for?"

GENERAL GUIDELINES:
7. When sharing specs, clearly indicate which source document each fact comes from.
8. If the user asks to "show" or "display" a data sheet, explain that you can provide the key specifications and information FROM the data sheet, but cannot display the original PDF.
9. When sharing product specifications, organize them clearly with the product name, key features, and technical specs.
10. If you genuinely cannot find specific information in the context, still be helpful — share what you DO know and suggest what the user might search for next.

CONTEXT FROM PRODUCT DOCUMENTS:
{context}"""

st.set_page_config(page_title="EV/Dynacord Chatbot", page_icon="🔊")
st.title("🔊 Electro-Voice / Dynacord Chatbot")
st.caption("Ask about any EV or Dynacord product — specs, setup, troubleshooting, and more.")

# Load API key from secrets (auto) or sidebar (manual fallback)
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
else:
    st.sidebar.success("✅ AI-powered answers enabled")

brand_filter = st.sidebar.selectbox("Filter by brand", ["All", "Electro-Voice", "Dynacord"])

# Load the configured retrieval backend
if RETRIEVAL_MODE == "chroma":
    store = load_chroma()
    st.sidebar.info(f"📚 {store.count():,} chunks (dense embeddings)")
else:
    vectorizer, tfidf_matrix, meta = load_tfidf()
    store = (vectorizer, tfidf_matrix, meta)
    st.sidebar.info(f"📚 {len(meta['texts']):,} chunks (TF-IDF)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about EV/Dynacord products..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    brand = None if brand_filter == "All" else brand_filter
    results = search(prompt, store, api_key=api_key, brand=brand, top_k=15)

    # Build rich context with document sources clearly labeled
    context_parts = []
    for i, r in enumerate(results):
        source = r['metadata'].get('filename', 'Unknown')
        brand_name = r['metadata'].get('brand', 'Unknown')
        context_parts.append(f"--- Document {i+1}: {source} ({brand_name}) ---\n{r['text']}")
    context = "\n\n".join(context_parts)

    sources = get_unique_sources(results)

    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Build conversation history for multi-turn context
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
        # Include recent conversation history (last 6 messages)
        for msg in st.session_state.messages[-7:-1]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
        )
        answer = response.choices[0].message.content

        # Add source references
        if sources:
            answer += "\n\n---\n📄 **Sources:** " + ", ".join(sources[:5])
    else:
        answer = "**Relevant documents found:**\n\n"
        for r in results[:10]:
            score_pct = f"{r['score']:.0%}"
            answer += f"📄 **{r['metadata']['filename']}** ({r['metadata']['brand']}) — relevance: {score_pct}\n{r['text'][:300]}...\n\n"
        answer += "\n*Add your OpenAI API key in Streamlit secrets or the sidebar for AI-powered answers.*"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
