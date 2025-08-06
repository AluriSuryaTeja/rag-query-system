import streamlit as st
import tempfile
import requests
from rag_utils import run_rag_pipeline

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("RAG-based Insurance Assistant")

option = st.radio("How would you like to upload the PDF?", ("Upload from device", "Provide PDF URL"))

# --- Upload PDF ---
pdf_file = None
if option == "Upload from device":
    pdf_file = st.file_uploader("Upload your insurance document (PDF)", type=["pdf"])
else:
    pdf_url = st.text_input("Paste the URL to the PDF file:")
    if pdf_url:
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(response.content)
                pdf_file = open(tmp_file.name, "rb")
        except Exception as e:
            st.error(f"Failed to download PDF: {e}")

# --- Input Questions ---
questions_input = st.text_area("Enter your questions (comma-separated):", height=150)

# --- Process Button ---
if st.button("Run RAG Pipeline"):
    if not pdf_file or not questions_input.strip():
        st.error("Please upload a PDF and enter at least one question.")
    else:
        try:
            # Prepare questions
            questions = [q.strip() for q in questions_input.split(",") if q.strip()]
            st.info("Processing document and generating answers...")

            # Run RAG pipeline
            answers = run_rag_pipeline(pdf_file, questions)

            # Display results
            st.success("Answers generated:")
            for idx, ans in enumerate(answers):
                st.markdown(f"**Q{idx+1}:** {questions[idx]}")
                st.markdown(f"**A{idx+1}:** {ans}")
                st.markdown("---")

        except Exception as e:
            st.error(f"Error during processing: {e}")