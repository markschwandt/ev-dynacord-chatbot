import os
import json
import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

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

def search(query, vectorizer, tfidf_matrix, meta, brand=None, top_k=5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    if brand:
        for i, m in enumerate(meta["metadatas"]):
            if m["brand"] != brand:
                scores[i] = 0
    top_indices = scores.argsort()[-top_k:][::-1]
    return [
        {"text": meta["texts"][idx], "metadata": meta["metadatas"][idx], "score": float(scores[idx])}
        for idx in top_indices if scores[idx] > 0
    ]

st.set_page_config(page_title="EV/Dynacord Chatbot", page_icon="🔊")
st.title("🔊 Electro-Voice / Dynacord Chatbot")

# Load API key from secrets (auto) or sidebar (manual fallback)
api_key = st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
else:
    st.sidebar.success("OpenAI API key loaded from secrets")

brand_filter = st.sidebar.selectbox("Filter by brand", ["All", "Electro-Voice", "Dynacord"])

vectorizer, tfidf_matrix, meta = load_vectorstore()
st.sidebar.success(f"Loaded {len(meta['texts']):,} document chunks")

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
    results = search(prompt, vectorizer, tfidf_matrix, meta, brand=brand)
    context = "\n\n".join([f"[{r['metadata']['filename']}]: {r['text']}" for r in results])

    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant for Electro-Voice and Dynacord professional audio products. Answer based on these documents:\n\n{context}"},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response.choices[0].message.content
    else:
        answer = "**Relevant documents found:**\n\n"
        for r in results:
            answer += f"📄 **{r['metadata']['filename']}** ({r['metadata']['brand']})\n{r['text']}\n\n"
        answer += "\n*Add your OpenAI API key in Streamlit secrets or the sidebar for AI-powered answers.*"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
