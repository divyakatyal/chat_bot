from flask import Flask, request, jsonify
import openai
import os
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

DATA_PATH = "Training Dataset.csv"
df = pd.read_csv(DATA_PATH).fillna("")
documents = df.apply(lambda row: " | ".join(row.astype(str)), axis=1).tolist()

# Load embedding model and index
embedder = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedder.encode(documents, show_progress_bar=True)

dimension = document_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings).astype('float32'))

def retrieve_context(question, k=3):
    query_embedding = embedder.encode([question])
    D, I = index.search(np.array(query_embedding).astype('float32'), k)
    return "\n".join([documents[i] for i in I[0]])

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        context = retrieve_context(question)
        prompt = f"""You are a helpful assistant. Use the following context from a loan application dataset to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
