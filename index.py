import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"

# 🔥 LOCAL EMBEDDINGS (NO API CALLS)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Idempotent
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

all_docs = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file
            doc.metadata["category"] = file.split("_")[0]

        all_docs.extend(docs)

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

split_docs = splitter.split_documents(all_docs)

# Store
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=PERSIST_DIR
)
print("✅ Index built (NO API used)")