import uuid
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict

from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# CONFIG
# =========================
PERSIST_DIR = "chroma_db"

# ✅ LOCAL EMBEDDINGS (NO API CALL)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ LLM (ONLY API CALL)
import os

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  # ✅ use env
    api_version=os.getenv("OPENAI_API_VERSION"),
    temperature=0
)

# ✅ VECTOR STORE
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding
)

# =========================
# FASTAPI INIT
# =========================
app = FastAPI()

# In-memory chat storage
chat_memory: Dict[str, List[str]] = {}

# =========================
# REQUEST / RESPONSE MODELS
# =========================
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class Source(BaseModel):
    document: str
    page: int
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: str
    conversation_id: str


# =========================
# PROMPT
# =========================
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Gapblue policy assistant.
Answer ONLY from the provided context.
Cite sources in your answer like (Source: file, Page: X).
If the answer is not in the context, say: "I don't know." """),
    ("human", """
Context:
{context}

History:
{history}

Question:
{question}
""")
])

# =========================
# HELPERS
# =========================
def format_docs(docs):
    return "\n\n".join([
        f"{d.page_content}\n(Source: {d.metadata.get('source')}, Page: {d.metadata.get('page')})"
        for d in docs
    ])


def get_confidence(scores):
    if not scores:
        return "low"

    avg = sum(scores) / len(scores)

    # Chroma returns distance → lower is better
    if avg < 0.3:
        return "high"
    elif avg < 0.6:
        return "medium"
    return "low"


# =========================
# API: /chat
# =========================
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    # Create or reuse conversation ID
    cid = req.conversation_id or str(uuid.uuid4())

    if cid not in chat_memory:
        chat_memory[cid] = []

    history = "\n".join(chat_memory[cid][-4:])

    # ✅ Retrieval (NO API CALL)
    docs_with_scores = vectorstore.similarity_search_with_score(req.question, k=4)

    docs = []
    scores = []

    for d, s in docs_with_scores:
        d.metadata["score"] = float(s)
        docs.append(d)
        scores.append(float(s))

    # ✅ Handle no results
    if not docs:
        return {
            "answer": "I don't know based on the provided context.",
            "sources": [],
            "confidence": "low",
            "conversation_id": cid
        }

    context = format_docs(docs)

    # ✅ ONLY LLM CALL
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "history": history,
        "question": req.question
    })

    answer = response.content

    # Save memory
    chat_memory[cid].append(f"Q: {req.question}")
    chat_memory[cid].append(f"A: {answer}")

    # Format sources
    sources = [
        {
            "document": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "relevance_score": round(d.metadata.get("score", 0.0), 3)
        }
        for d in docs
    ]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": get_confidence(scores),
        "conversation_id": cid
    }


# =========================
# API: /sources
# =========================
@app.get("/sources")
def get_sources():
    collection = vectorstore._collection.get()

    source_counts = {}

    for meta in collection["metadatas"]:
        src = meta.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "documents": [
            {"document": k, "chunk_count": v}
            for k, v in source_counts.items()
        ]
    }