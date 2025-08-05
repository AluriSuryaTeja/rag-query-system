import os
import faiss
import requests
import numpy as np
# import openai
# from openai import OpenAI
from pypdf import PdfReader
# from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ===================== ENV SETUP ===================== #
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
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

def generate_answer(query, context, model="llama3.2"):

    url = "http://localhost:11434/api/chat"


    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an insurance assistant. Provide short, concise answers (2–3 sentences max) based strictly on relevant clauses.. You are given input in a JSON Format as { <Question> : <Context> }. Give the output in JSON format as {<AnswerNumber> : <Answer> }"},
            {"role": "user", "content": str(context)}
        ],
        "stream": False
    }

    response = requests.post(url , json = payload)
    try:
        data = response.json()
        if "message" in data:
            return data["message"]["content"]
        else:
            print("❌ Unexpected API response:", data)
            return f"Error: unexpected response format: {data}"
    except Exception as e:
        print("❌ Failed to parse response:", response.text)
        return f"Error: failed to parse response - {str(e)}"

    


# ===================== MAIN FLOW ===================== #

def run_rag_pipeline_from_chunks(pdf_path, questions):
    print("📄 Loading document...")
    chunks = load_pdf_chunks(pdf_path)

    print("🔐 Creating embeddings...")
    embeddings = embed_chunks(chunks)

    print("📦 Creating FAISS index...")
    index = create_faiss_index(embeddings)

    print("🔍 Processing questions and generating answers...")

    allQueries = {}
    for question in questions:
        relevant = search_index(question, chunks, index)
        full_context = "\n---\n".join(relevant)
        allQueries[question] = full_context
    answer = generate_answer(question, allQueries)
    # answers = 'GENERATED ANSWER'
    

    return answer

# ===================== EXAMPLE USAGE ===================== #
if __name__ == "__main__":
    user_pdf = "data/sample.pdf"  # Or ask for input()
    user_query = user_queries = [
      "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
      "What is the waiting period for pre-existing diseases (PED) to be covered?",
      "Does this policy cover maternity expenses, and what are the conditions?",
      "What is the waiting period for cataract surgery?",
      "Are the medical expenses for an organ donor covered under this policy?",
      "What is the No Claim Discount (NCD) offered in this policy?",
      "Is there a benefit for preventive health check-ups?",
      "How does the policy define a 'Hospital'?",
      "What is the extent of coverage for AYUSH treatments?",
      "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    answers = run_rag_pipeline_from_chunks(user_pdf, user_query)
    print(answers)