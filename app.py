from flask import Flask, request, jsonify
import os
import tempfile
import requests
from rag_utils import run_rag_pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)  # Corrected: __name__ instead of _name_

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    data = request.get_json()

    pdf_url = data.get("documents")
    questions = data.get("questions")

    if not pdf_url or not questions:
        return jsonify({"error": "Missing 'documents' or 'questions' field."}), 400

    try:
        # Download the PDF temporarily
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        # Process the questions
        answers = run_rag_pipeline(tmp_file_path, questions)
        return jsonify({"answers": answers})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":  # Corrected
    app.run(debug=True, host="0.0.0.0", port=5000)