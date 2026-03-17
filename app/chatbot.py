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

