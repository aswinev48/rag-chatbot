# 📌 RAG Chatbot – Gapblue Policy Assistant

---

## 🧠 Architecture

User Query  
   ↓  
FastAPI (/chat)  
   ↓  
Retriever (ChromaDB)  
   ↓  
Relevant Chunks (with metadata)  
   ↓  
LLM (Azure OpenAI)  
   ↓  
Response + Sources + Confidence  

---

## ⚙️ Tech Stack

- FastAPI
- LangChain
- Azure OpenAI
- Sentence Transformers (local embeddings)
- ChromaDB
- Docker

---

## 🚀 Setup Instructions

### 1. Clone Repo

```bash
git clone <your-repo-url>
cd rag_chatbot