import os
import json
import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vectorstore")

@st.cache_resource
def load_vectorstore():
    with open(os.path.join(VECTORSTORE_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(VECTORSTORE_DIR, "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(os.path.join(VECTORSTORE_DIR, "chunks_meta.json"), "r") as f:
        meta = json.load(f)
    return vectorizer, tfidf_matrix, meta

def search(query, vectorizer, tfidf_matrix, meta, brand=None, top_k=15):
    """Multi-strategy search: tries original query, then expanded terms."""
    query_lower = query.lower()

    # Strategy 1: Direct TF-IDF search on original query
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Strategy 2: Also search for individual key terms to catch product names
    product_terms = re.findall(r'[A-Za-z]+[\s\-]?\d+[A-Za-z]*|\b[A-Z]{2,}[- ]?\d+\w*', query)
    if product_terms:
        for term in product_terms:
            term_vec = vectorizer.transform([term])
            term_scores = cosine_similarity(term_vec, tfidf_matrix).flatten()
            scores = np.maximum(scores, term_scores * 0.9)

    # Strategy 3: Also check filename matches (very reliable for product lookups)
    for i, m in enumerate(meta["metadatas"]):
        fname = m.get("filename", "").lower()
        for word in query_lower.split():
            if len(word) > 2 and word in fname:
                scores[i] = max(scores[i], 0.3)

    # Apply brand filter
    if brand:
        for i, m in enumerate(meta["metadatas"]):
            if m.get("brand", "") != brand:
                scores[i] = 0

    top_indices = scores.argsort()[-top_k:][::-1]
    results = [
        {"text": meta["texts"][idx], "metadata": meta["metadatas"][idx], "score": float(scores[idx])}
        for idx in top_indices if scores[idx] > 0
    ]
    return results

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

SYSTEM_PROMPT = """You are a knowledgeable and helpful product support assistant for Electro-Voice (EV) and Dynacord professional audio equipment. You have access to a large library of product documentation including spec sheets, user guides, engineering data sheets, and technical documents.

IMPORTANT GUIDELINES:
1. ALWAYS be helpful. If documents are provided as context below, USE them to answer the question as thoroughly as possible.
2. NEVER say "I don't have access to" or "I don't have information about" a product unless the context truly contains zero relevant information. The context below comes from a database of over 60,000 document chunks - if relevant docs were found, trust them.
3. When the user asks about a specific product (like "EVERSE 8" or "EKX-15P"), look carefully through ALL the provided context chunks. The product info IS likely there.
4. If the user asks to "show" or "display" a data sheet, explain that you can provide the key specifications and information FROM the data sheet, but cannot display the original PDF. Then provide the relevant specs.
5. When sharing product specifications, organize them clearly with the product name, key features, and technical specs.
6. If you genuinely cannot find specific information in the context, still be helpful - share what you DO know from the context and suggest what the user might search for next.
7. Always mention which document(s) your information comes from.

CONTEXT FROM PRODUCT DOCUMENTS:
{context}"""

st.set_page_config(page_title="EV/Dynacord Chatbot", page_icon="🔊")
st.title("🔊 Electro-Voice / Dynacord Chatbot")
st.caption("Ask about any EV or Dynacord product - specs, setup, troubleshooting, and more.")

# Load API key from secrets (auto) or sidebar (manual fallback)
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
else:
    st.sidebar.success("AI-powered answers enabled")

brand_filter = st.sidebar.selectbox("Filter by brand", ["All", "Electro-Voice", "Dynacord"])

vectorizer, tfidf_matrix, meta = load_vectorstore()
st.sidebar.info(f"Library: {len(meta['texts']):,} document chunks loaded")

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
    results = search(prompt, vectorizer, tfidf_matrix, meta, brand=brand, top_k=15)

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
            temperature=0.3,
        )
        answer = response.choices[0].message.content

        # Add source references
        if sources:
            answer += "\n\n---\nSources: " + ", ".join(sources[:5])
    else:
        answer = "**Relevant documents found:**\n\n"
        for r in results[:10]:
            score_pct = f"{r['score']:.0%}"
            answer += f"**{r['metadata']['filename']}** ({r['metadata']['brand']}) - relevance: {score_pct}\n{r['text'][:300]}...\n\n"
        answer += "\n*Add your OpenAI API key in Streamlit secrets or the sidebar for AI-powered answers.*"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
