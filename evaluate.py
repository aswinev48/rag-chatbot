import json
import requests
from datetime import datetime

# =========================
# CONFIG
# =========================
API_URL = "http://127.0.0.1:8000/chat"   # use 127.0.0.1 explicitly
OUTPUT_FILE = "evaluation_results.json"

# =========================
# TEST DATA (20 QUESTIONS)
# =========================
test_cases = [
    # HR Policy
    {"question": "What is the purpose of the leave policy?", "expected": "Explains leave entitlement and procedure"},
    {"question": "How is casual leave deducted?", "expected": "Half-day deduction"},
    {"question": "Who is eligible for leave benefits?", "expected": "Employees"},
    {"question": "What happens if leave rules are violated?", "expected": "Penalty applies"},

    # POSH Policy
    {"question": "What does POSH policy stand for?", "expected": "Prevention of Sexual Harassment"},
    {"question": "Who does POSH policy protect?", "expected": "Employees"},
    {"question": "What kind of behavior is covered under POSH?", "expected": "Sexual harassment"},
    {"question": "What action can be taken under POSH?", "expected": "Complaint and investigation"},

    # Information Security Policy
    {"question": "What is the goal of information security policy?", "expected": "Protect company data"},
    {"question": "Who is responsible for data security?", "expected": "Employees"},
    {"question": "What should be done in case of security breach?", "expected": "Report immediately"},
    {"question": "What kind of data must be protected?", "expected": "Confidential data"},

    # Financial Support Policy
    {"question": "What does financial support policy provide?", "expected": "Financial assistance"},
    {"question": "Who can avail financial support?", "expected": "Employees"},
    {"question": "What types of expenses are covered?", "expected": "Approved expenses"},
    {"question": "Is approval required for financial support?", "expected": "Yes"},

    # Anti-Corruption Policy
    {"question": "What does anti-corruption policy prohibit?", "expected": "Bribery"},
    {"question": "Who must follow anti-corruption policy?", "expected": "All employees"},
    {"question": "What should be done if corruption is suspected?", "expected": "Report it"},
    {"question": "Does policy allow gifts or bribes?", "expected": "No"}
]

# =========================
# MANUAL RATING FUNCTION
# =========================
def get_manual_rating(question, answer):
    print("\n---------------------------")
    print(f"Q: {question}")
    print(f"A: {answer}")
    print("Rate (correct / partial / wrong / hallucinated): ", end="")
    return input().strip().lower()


# =========================
# RUN EVALUATION
# =========================
results = []

for test in test_cases:
    try:
        response = requests.post(
            API_URL,
            json={"question": test["question"]},
            timeout=30
        )

        print("\n🔍 Status Code:", response.status_code)

        if response.status_code != 200:
            print("❌ API Error:", response.text)
            raise Exception("API returned non-200 status")

        try:
            data = response.json()
        except Exception:
            print("❌ JSON Parse Error:", response.text)
            raise Exception("Invalid JSON response")

        answer = data.get("answer", "")
        sources = data.get("sources", [])

    except Exception as e:
        print(f"\n❌ Error for question: {test['question']}")
        print("Error details:", str(e))

        answer = "ERROR"
        sources = []

    rating = get_manual_rating(test["question"], answer)

    results.append({
        "question": test["question"],
        "expected": test["expected"],
        "answer": answer,
        "sources": sources,
        "rating": rating
    })


# =========================
# METRICS CALCULATION
# =========================
total = len(results)

correct = sum(1 for r in results if r["rating"] == "correct")
partial = sum(1 for r in results if r["rating"] == "partial")
wrong = sum(1 for r in results if r["rating"] == "wrong")
hallucinated = sum(1 for r in results if r["rating"] == "hallucinated")

avg_sources = sum(len(r["sources"]) for r in results) / total if total > 0 else 0

accuracy = correct / total if total > 0 else 0
hallucination_rate = hallucinated / total if total > 0 else 0

summary = {
    "total_questions": total,
    "correct": correct,
    "partial": partial,
    "wrong": wrong,
    "hallucinated": hallucinated,
    "accuracy_rate": round(accuracy, 2),
    "hallucination_rate": round(hallucination_rate, 2),
    "average_sources_per_answer": round(avg_sources, 2),
    "timestamp": datetime.now().isoformat()
}

# =========================
# SAVE RESULTS
# =========================
output = {
    "summary": summary,
    "details": results
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=4)

# =========================
# PRINT SUMMARY TABLE
# =========================
print("\n📊 EVALUATION SUMMARY")
print("-----------------------------")
print(f"Total Questions: {summary['total_questions']}")
print(f"Correct: {summary['correct']}")
print(f"Partial: {summary['partial']}")
print(f"Wrong: {summary['wrong']}")
print(f"Hallucinated: {summary['hallucinated']}")
print(f"Accuracy Rate: {summary['accuracy_rate']}")
print(f"Hallucination Rate: {summary['hallucination_rate']}")
print(f"Avg Sources/Answer: {summary['average_sources_per_answer']}")

print("\n✅ Evaluation Complete!")