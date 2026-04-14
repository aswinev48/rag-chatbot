from fastapi import FastAPI
from pydantic import BaseModel

from app.rag_chain import get_answer
from app.memory import get_history, update_history

app = FastAPI()


class ChatRequest(BaseModel):
    question: str
    conversation_id: str = "default"


@app.post("/chat")
def chat(req: ChatRequest):

    history = get_history(req.conversation_id)

    answer, docs = get_answer(req.question, history)

    update_history(req.conversation_id, req.question, answer)

    sources = []
    for d in docs:
        sources.append({
            "document": d.metadata["source"],
            "page": d.metadata["page"]
        })

    return {
        "answer": answer,
        "sources": sources,
        "confidence": "high" if sources else "low"
    }


@app.get("/sources")
def sources():
    import os
    return {"documents": os.listdir("data")}