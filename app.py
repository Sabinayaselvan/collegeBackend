from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import csv

app = Flask(__name__)
CORS(app)

qa_data = {}

def load_csv_data():
    global qa_data
    qa_data = {}
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

load_csv_data()

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()
    if not user_input:
        return jsonify({"response": "Please ask a question."}), 400

    # Exact match
    if user_input in qa_data:
        return jsonify({"response": qa_data[user_input]})

    # Partial match
    for question, answer in qa_data.items():
        if question in user_input or user_input in question:
            return jsonify({"response": answer})

    return jsonify({"response": "Sorry, I don't have an answer for that yet."})

@app.route("/")
def home():
    return "CollegeBot Flask Server is running!", 200

@app.route("/ping")
def ping():
    return "pong", 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
