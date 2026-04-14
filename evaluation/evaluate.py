import requests
import json

with open("evaluation/questions.json") as f:
    questions = json.load(f)

results = []

for q in questions:
    res = requests.post(
        "http://localhost:8000/chat",
        json={"question": q["question"], "conversation_id": "eval"}
    ).json()

    results.append({
        "question": q["question"],
        "answer": res["answer"],
        "sources": res["sources"],
        "rating": ""  # fill manually
    })

with open("evaluation/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Evaluation done")