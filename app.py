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
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) > 1:
                    header = [h.lower().strip() for h in rows[0]]
                    try:
                        q_index = header.index("question")
                        a_index = header.index("answer")
                    except ValueError:
                        continue  # skip file if header is incorrect
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

    # Exact match
    if user_input in qa_data:
        return jsonify({"response": qa_data[user_input]})

    # Partial match
    for question, answer in qa_data.items():
        if question in user_input or user_input in question:
            return jsonify({"response": answer})

    return jsonify({"response": "Sorry, I don't have an answer for that yet."})

@app.route("/ping")
def ping():
    return "pong", 200

if __name__ == "__main__":
    app.run(debug=True)
