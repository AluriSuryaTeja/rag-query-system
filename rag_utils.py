import os
import faiss
import numpy as np
import openai
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ===================== ENV SETUP ===================== #
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===================== UTILITY FUNCTIONS ===================== #
def load_pdf_chunks(pdf, chunk_min_len=50):
    if hasattr(pdf, "read"):
        reader = PdfReader(pdf)
    else:
        reader = PdfReader(open(pdf, "rb"))

    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) >= chunk_min_len:
                    chunks.append(para)
    return chunks

def embed_chunks(chunks):
    return embedding_model.encode(chunks, convert_to_tensor=False)

def create_faiss_index(vectors):
    dim = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))
    return index

def search_index(query, chunks, index, k=3):
    query_vec = embedding_model.encode([query])[0]
    D, I = index.search(np.array([query_vec]), k)
    return [chunks[i] for i in I[0]]

def generate_answer(query, context, model="gpt-4"):
    client = OpenAI()
    messages = [
        {"role": "system", "content": "You are an insurance assistant. Provide short, concise answers (2–3 sentences max) based strictly on relevant clauses."},
        {"role": "user", "content": f"Query: {query}\n\nRelevant Clauses:\n{context}"}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=150  # ✅ Limits the response length
    )
    return response.choices[0].message.content.strip()


# ===================== MAIN FLOW ===================== #

def run_rag_pipeline_from_chunks(pdf_path, questions):
    print("📄 Loading document...")
    chunks = load_pdf_chunks(pdf_path)

    print("🔐 Creating embeddings...")
    embeddings = embed_chunks(chunks)

    print("📦 Creating FAISS index...")
    index = create_faiss_index(embeddings)

    print("🔍 Processing questions and generating answers...")
    answers = []
    for question in questions:
        relevant = search_index(question, chunks, index)
        full_context = "\n---\n".join(relevant)
        answer = generate_answer(question, full_context)
        answers.append(answer)

    return answers

# ===================== EXAMPLE USAGE ===================== #
if __name__ == "__main__":
    user_pdf = "data/sample.pdf"  # Or ask for input()
    user_query = input("❓ Enter your question: ")
    run_rag_pipeline(user_pdf, user_query)