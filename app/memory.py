from langchain.memory import ConversationBufferMemory

memory_store = {}

def get_memory(conversation_id: str):
    if conversation_id not in memory_store:
        memory_store[conversation_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return memory_store[conversation_id]