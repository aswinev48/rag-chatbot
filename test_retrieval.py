from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

query = "What is the leave policy?"

docs = retriever.invoke(query)

for i, d in enumerate(docs):
    print(f"\nResult {i+1}:")
    print("Content:", d.page_content[:200])
    print("Metadata:", d.metadata)