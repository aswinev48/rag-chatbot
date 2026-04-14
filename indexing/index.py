import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data"
PERSIST_DIR = "chroma_db"

def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            pages = loader.load()

            for p in pages:
                p.metadata["source"] = file
                p.metadata["page"] = p.metadata.get("page", 0)
                p.metadata["category"] = file.replace(".pdf", "")

            docs.extend(pages)
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)

def build_index():
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    docs = load_documents()
    chunks = split_docs(docs)

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    vectordb.persist()
    print("✅ Index built!")

if __name__ == "__main__":
    build_index()