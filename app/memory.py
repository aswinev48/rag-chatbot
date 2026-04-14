memory_store = {}

def get_history(conversation_id):
    if conversation_id not in memory_store:
        memory_store[conversation_id] = []
    return memory_store[conversation_id]

def update_history(conversation_id, question, answer):
    history = memory_store[conversation_id]

    history.append(f"User: {question}")
    history.append(f"Assistant: {answer}")

    # 🔥 keep last 2 turns only
    memory_store[conversation_id] = history[-4:]