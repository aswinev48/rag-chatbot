import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = "chroma_db"

# ✅ LOCAL embeddings (same as index)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# ✅ Azure LLM (only API usage now)
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    temperature=0
)


def get_answer(question, chat_history):

    docs = retriever.invoke(question)

    if not docs:
        return "I don't know based on the provided documents.", []

    context = "\n\n".join([
        f"{d.page_content}\n(Source: {d.metadata['source']}, Page: {d.metadata['page']})"
        for d in docs
    ])

    history_text = "\n".join(chat_history[-4:])

    prompt = f"""
You are a Gapblue policy assistant.

- Answer ONLY from the context
- Cite sources
- If unsure, say "I don't know"

Chat History:
{history_text}

Context:
{context[:1500]}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content, docs