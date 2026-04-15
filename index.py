import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
PERSIST_DIRECTORY = "chroma_db_gapblue_v4"

CATEGORY_MAP = {
    "anti_corruption_policy.pdf": "compliance",
    "information_security_policy.pdf": "it_security",
    "financial_support_policy.pdf": "finance_policy",
    "hr_policy_manual.pdf": "hr",
    "posh_policy.pdf": "posh_policy"
}


def load_documents():
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    all_docs = []

    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_DIR, pdf_file)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = pdf_file

        all_docs.extend(docs)

    return all_docs


def split_documents(all_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(all_docs)


def enrich_metadata(chunks):
    for chunk in chunks:
        source_file = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page", 0)

        chunk.metadata["document"] = source_file
        chunk.metadata["page_number"] = page + 1
        chunk.metadata["category"] = CATEGORY_MAP.get(source_file, "other")

    return chunks


def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vectorstore(chunks, embedding_model):
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )


def main():
    print("Loading documents...")
    all_docs = load_documents()
    print(f"Total pages loaded: {len(all_docs)}")

    print("Splitting documents...")
    chunks = split_documents(all_docs)
    print(f"Total chunks created: {len(chunks)}")

    print("Adding metadata...")
    chunks = enrich_metadata(chunks)

    print("Loading embedding model...")
    embedding_model = create_embedding_model()

    print("Creating Chroma DB...")
    build_vectorstore(chunks, embedding_model)

    print("Indexing complete!")


if __name__ == "__main__":
    main()