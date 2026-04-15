import requests
import json
from tabulate import tabulate

API_URL = "http://127.0.0.1:8000/chat"

# =========================
# TEST QUESTIONS (20 TOTAL)
# =========================
test_data = [
    # HR POLICY
    {"question": "What are the types of leaves?", "expected": "Earned Leave, Personal Leave, etc."},
    {"question": "How many earned leaves are allowed?", "expected": "12 days"},
    {"question": "What is the dress code for men?", "expected": "Formal wear"},
    {"question": "What are working hours?", "expected": "10 AM to 7 PM"},

    # IT SECURITY
    {"question": "What is the purpose of information security policy?", "expected": "Protect data"},
    {"question": "Can employees share confidential data?", "expected": "No"},
    {"question": "What happens if data is leaked?", "expected": "Disciplinary action"},
    {"question": "What is confidential information?", "expected": "Sensitive company data"},

    # FINANCE
    {"question": "What is financial support policy?", "expected": "Reimbursement policy"},
    {"question": "What is required for reimbursement?", "expected": "Bills"},
    {"question": "Who approves claims?", "expected": "Finance team"},
    {"question": "What happens for false claims?", "expected": "Disciplinary action"},

    # ASSET
    {"question": "What is asset management policy?", "expected": "Manage assets"},
    {"question": "What should employees do with assets?", "expected": "Use responsibly"},
    {"question": "What happens if assets are lost?", "expected": "Employee liable"},
    {"question": "When to return assets?", "expected": "Exit time"},

    # ANTI-CORRUPTION
    {"question": "What is anti-corruption policy?", "expected": "Prevent bribery"},
    {"question": "Can employees accept gifts?", "expected": "No"},
    {"question": "What is corruption?", "expected": "Bribery"},
    {"question": "What happens if violated?", "expected": "Legal action"},
]

# =========================
# CALL API
# =========================
def ask_question(question):
    try:
        response = requests.post(API_URL, json={"question": question})
        data = response.json()

        return {
            "answer": data.get("answer", ""),
            "sources": data.get("sources", [])
        }

    except Exception as e:
        return {
            "answer": f"ERROR: {str(e)}",
            "sources": []
        }


# =========================
# RUN EVALUATION
# =========================
def run_evaluation():
    results = []

    print("Running evaluation...\n")

    for item in test_data:
        question = item["question"]
        expected = item["expected"]

        print(f"Q: {question}")

        response = ask_question(question)

        result = {
            "question": question,
            "expected_answer": expected,
            "model_answer": response["answer"],
            "sources": response["sources"],
            "rating": ""  # to fill manually
        }

        results.append(result)

    # Save raw results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Results saved to evaluation_results.json")
    print("👉 Now manually add ratings (correct/partial/wrong/hallucinated)\n")


# =========================
# METRICS
# =========================
def compute_metrics():
    with open("evaluation_results.json", "r") as f:
        data = json.load(f)

    total = len(data)
    correct = sum(1 for d in data if d["rating"] == "correct")
    hallucinated = sum(1 for d in data if d["rating"] == "hallucinated")
    total_sources = sum(len(d["sources"]) for d in data)

    accuracy = correct / total if total else 0
    hallucination_rate = hallucinated / total if total else 0
    avg_sources = total_sources / total if total else 0

    print("\n📊 METRICS")
    print(f"Total Questions: {total}")
    print(f"Accuracy Rate: {accuracy:.2f}")
    print(f"Hallucination Rate: {hallucination_rate:.2f}")
    print(f"Average Sources per Answer: {avg_sources:.2f}")

    # Summary Table
    table = []
    for d in data:
        table.append([
            d["question"],
            d["rating"],
            len(d["sources"])
        ])

    print("\n📋 SUMMARY TABLE")
    print(tabulate(table, headers=["Question", "Rating", "Sources"], tablefmt="grid"))


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("1. Run evaluation")
    print("2. Compute metrics")
    choice = input("Enter choice (1/2): ")

    if choice == "1":
        run_evaluation()
    elif choice == "2":
        compute_metrics()