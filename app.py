import streamlit as st
from rag_utils import *

st.set_page_config(page_title="Insurance RAG Assistant", layout="wide")

st.title("Insurance Clause Assistant (RAG-based)")

uploaded_file = st.file_uploader("Upload Insurance PDF", type=["pdf"])

if uploaded_file:
    query = st.text_input("Ask a question about the uploaded document")

    if query:
        with st.spinner("Processing..."):
            chunks = load_pdf_chunks(uploaded_file)
            embeddings = embed_chunks(chunks)
            index = create_faiss_index(embeddings)
            relevant = search_index(query, chunks, index)
            context = "\n---\n".join(relevant)
            answer = generate_answer(query, context)

        st.success("Answer:")
        st.markdown(answer)