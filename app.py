from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import csv
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

qa_data = {}
qa_embeddings = []

# ✅ Correct environment variable name
HF_API_KEY = os.environ.get("HF_API_KEY")

def get_embedding_from_hf(text):
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json={"inputs": text})
        if response.status_code == 200:
            embedding = response.json()
            # Some models return nested list: [[...]], others: [...]
            if isinstance(embedding[0], list):
                return np.array(embedding).reshape(1, -1)
            else:
                return np.array([embedding])
        else:
            print("❌ Hugging Face API error:", response.status_code)
            print(response.text)
            return np.zeros((1, 384))  # fallback
    except Exception as e:
        print("❌ Exception during embedding:", str(e))
        return np.zeros((1, 384))  # fallback


def load_csv_data():
    global qa_data, qa_embeddings
    qa_data = {}
    qa_embeddings = []
    data_folder = "data"

    if not os.path.exists(data_folder):
        print(f"❌ Data folder '{data_folder}' not found.")
        return

    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_folder, filename)
            encodings_to_try = ["utf-8", "ISO-8859-1", "latin1"]
            rows = []

            for enc in encodings_to_try:
                try:
                    with open(file_path, "r", newline='', encoding=enc) as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                    print(f"✅ Loaded {filename} using {enc}")
                    break
                except UnicodeDecodeError:
                    print(f"⚠️ {enc} failed for {filename}, trying next...")
                except Exception as e:
                    print(f"❌ Error reading {filename} with {enc}: {e}")

            if not rows:
                print(f"❌ Skipping {filename} — failed to decode.")
                continue

            header = [h.lower().strip() for h in rows[0]]
            try:
                q_index = header.index("question")
                a_index = header.index("answer")
            except ValueError:
                print(f"❌ Skipping {filename} — missing 'question' or 'answer' columns.")
                continue

            for row in rows[1:]:
                if len(row) > max(q_index, a_index):
                    question = row[q_index].strip().lower()
                    answer = row[a_index].strip()
                    if question and answer:
                        qa_data[question] = answer
                        embedding = get_embedding_from_hf(question)
                        qa_embeddings.append((question, answer, embedding))

load_csv_data()

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()
    if not user_input:
        return jsonify({"response": "Please ask a question."}), 400

    # ✅ Exact match
    if user_input in qa_data:
        return jsonify({"response": qa_data[user_input]})

    # ✅ Partial match
    for question, answer in qa_data.items():
        if question in user_input or user_input in question:
            return jsonify({"response": answer})

    # ✅ Semantic match via Hugging Face
    try:
        user_embedding = get_embedding_from_hf(user_input)
    except Exception as e:
        return jsonify({"response": f"Embedding error: {str(e)}"}), 500

    best_score = 0.0
    best_answer = None

    for question, answer, embedding in qa_embeddings:
        score = cosine_similarity(user_embedding, embedding)[0][0]
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_score > 0.6:
        return jsonify({"response": best_answer})

    return jsonify({"response": "Sorry, I don't have an answer for that yet."})


@app.route("/")
def home():
    return "CollegeBot Flask Server is running!", 200

@app.route("/ping")
def ping():
    return "pong", 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
