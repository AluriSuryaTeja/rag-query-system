#  RAG Document Query System

A **Retrieval-Augmented Generation (RAG)** project that enables users to upload documents (**PDF, Word, TXT, URLs**) and ask questions related to the content.  
The system uses **Ollama models** for response generation and **Sentence Transformers** for embedding-based document retrieval.

* * *

## Features

*  **Multi-format document support** – Upload PDF, Word, TXT, or provide a document URL
    
*  **Efficient document search** – Uses **Sentence Transformers** to generate semantic embeddings for document chunks
    
*  **Generative Answers** – Powered by **Ollama models** for context-aware responses
    
*  **Fast Retrieval** – Embedding-based search ensures quick and relevant answers
        

* * *

## Architecture

1.  **Document Upload** → User uploads a document or URL
    
2.  **Text Extraction** → The system extracts and chunks the text
    
3.  **Embedding Generation** → Sentence Transformers create vector embeddings for chunks
    
4.  **Retriever** → Finds the most relevant chunks for the query
    
5.  **Ollama Model** → Generates final answer using retrieved context
    

* * *

##  Tech Stack

*   **Backend:** Python
    
*   **Models:** Ollama for LLM responses
    
*   **Embeddings:** Sentence Transformers
    
*   **Vector Store:** FAISS 
    
*   **Document Parsing:** PyPDF2 
    
*   **Frontend:** Streamlit
    

* * *

##  Installation

### Clone the repository


CopyEdit

`git clone https://github.com/AluriSuryaTeja/rag-doc-query.git cd rag-doc-query`

### Create a virtual environment


CopyEdit

`python -m venv venv source venv/bin/activate  # On Windows: venv\Scripts\activate`

### Install dependencies



CopyEdit

`pip install -r requirements.txt`

###  Run the app

`python app.py`

* * *
