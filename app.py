import os
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI

load_dotenv()

# ================= ENV =================
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

PERSIST_DIRECTORY = "chroma_db_gapblue_v4"

app = FastAPI(title="Gapblue Document Q&A Chatbot")

# ================= EMBEDDINGS =================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================= VECTOR DB =================
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model
)

# ================= LLM =================
llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0
)

# ================= MEMORY =================
conversation_memory = {}

# ================= MODELS =================
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class SourceItem(BaseModel):
    document: str
    page: int
    category: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    confidence: str
    conversation_id: str


# ================= HELPERS =================
def format_docs(docs):
    formatted = []

    for doc in docs:
        content = doc.page_content.strip()

        if not content or len(content) < 30:
            continue

        document = doc.metadata.get("document", "unknown")
        page = doc.metadata.get("page_number", "unknown")

        formatted.append(
            f"""
SOURCE: {document} (Page {page})
CONTENT:
{content[:1000]}
"""
        )

    return "\n\n".join(formatted)


def deduplicate_sources(docs):
    unique_sources = []
    seen = set()

    for doc in docs:
        source = (
            doc.metadata.get("document"),
            doc.metadata.get("page_number"),
            doc.metadata.get("category")
        )

        if source not in seen:
            seen.add(source)
            unique_sources.append(
                SourceItem(
                    document=doc.metadata.get("document"),
                    page=doc.metadata.get("page_number"),
                    category=doc.metadata.get("category"),
                    relevance_score=1.0
                )
            )

    return unique_sources


def retrieve_docs(question: str):
    docs = vectorstore.similarity_search(question, k=6)
    return docs[:4]


def get_confidence(n):
    if n >= 3:
        return "high"
    elif n == 2:
        return "medium"
    else:
        return "low"


# ================= API =================
@app.get("/sources")
def get_sources():
    collection = vectorstore._collection
    data = collection.get(include=["metadatas"])

    chunk_count = {}
    for metadata in data["metadatas"]:
        document = metadata.get("document", "unknown")
        chunk_count[document] = chunk_count.get(document, 0) + 1

    return {
        "documents": [
            {"document": doc, "chunk_count": count}
            for doc, count in chunk_count.items()
        ]
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        # ================= Conversation ID =================
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # ================= Query Enhancement =================
        enhanced_query = f"""
        {request.question}

        Related terms:
        leave policy, earned leave, casual leave, sick leave,
        employee leave entitlement, types of leaves
        """

        if "leave" in request.question.lower():
            enhanced_query += """
            earned leave EL casual leave CL sick leave SL
            types of leaves leave categories 12 days leave
            """

        # ================= Retrieval =================
        docs = retrieve_docs(enhanced_query)

        print("\n==== RETRIEVED DOCS ====")
        for d in docs:
            print(d.metadata.get("page_number"), d.page_content[:150])

        context = format_docs(docs)

        # ================= Memory =================
        history = conversation_memory.get(conversation_id, [])
        history_text = "\n".join(history[-4:])

        # ================= PROMPT (🔥 FINAL FIX) =================
        prompt = f"""
You are a Gapblue policy assistant.

Answer ONLY from the provided context.

IMPORTANT:
- If answer exists indirectly, infer it.
- Extract lists clearly (like types of leaves).
- Do NOT say "I don't know" if relevant data exists.
- Only say "I don't know" if no relevant info is present.

Conversation:
{history_text}

Context:
{context}

Question:
{request.question}
"""

        # ================= LLM =================
        response = llm.invoke(prompt)
        answer_text = response.content

        # ================= Save Memory =================
        conversation_memory.setdefault(conversation_id, [])
        conversation_memory[conversation_id].append(f"User: {request.question}")
        conversation_memory[conversation_id].append(f"Assistant: {answer_text}")

        sources = deduplicate_sources(docs)

        return ChatResponse(
            answer=answer_text,
            sources=sources,
            confidence=get_confidence(len(docs)),
            conversation_id=conversation_id
        )

    except Exception as e:
        print("🔥 ERROR:", str(e))

        return ChatResponse(
            answer="Error occurred while processing the request.",
            sources=[],
            confidence="low",
            conversation_id=request.conversation_id or "unknown"
        )