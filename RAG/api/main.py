import os
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

# Setup dependencies
from src.ingestion.loaders import load_documents
from src.ingestion.chunking import chunk_documents
from src.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.vectorstore.chroma_store import ChromaDBStore
from src.retrieval.retriever import VectorRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.llm.openai_provider import OpenAILLMProvider
from src.prompt.prompt_manager import PromptManager
from src.prompt.builder import PromptBuilder
from src.generation.evidence_validator import EvidenceValidator
from src.rag_engine import RAGEngine

app = FastAPI(title="Finance RAG API")

# Setup global instances (simplified for assignment)
embedding_provider = OpenAIEmbeddingProvider()
chroma_store = ChromaDBStore("./chroma_db", "finance_docs", embedding_provider)
vector_retriever = VectorRetriever(chroma_store)
bm25_retriever = BM25Retriever(chroma_store)
hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
reranker = CrossEncoderReranker(hybrid_retriever)

llm_provider = OpenAILLMProvider(temperature=0.1)
prompt_manager = PromptManager()
prompt_builder = PromptBuilder(prompt_manager)
evidence_validator = EvidenceValidator()

rag_engine = RAGEngine(reranker, llm_provider, prompt_builder, evidence_validator)

class AskRequest(BaseModel):
    question: str
    top_k: int = 4

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    os.makedirs("./data", exist_ok=True)
    all_chunks = []
    
    for file in files:
        file_path = f"./data/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        docs = load_documents(file_path)
        chunks = chunk_documents(docs)
        all_chunks.extend(chunks)
        
    chroma_store.add_documents(all_chunks)
    # Rebuild BM25 index
    bm25_retriever._build_index()
    
    return {"files": len(files), "chunks": len(all_chunks)}

@app.post("/ask")
def ask(request: AskRequest):
    # Adjust reranker top_n based on request
    reranker.top_n = request.top_k
    result = rag_engine.ask(request.question)
    return result

@app.get("/stats")
def stats():
    count = chroma_store.collection.count()
    return {
        "Collection name": chroma_store.collection_name,
        "Total chunks": count,
        "Embedding model": embedding_provider.model_name,
        "LLM model": llm_provider.llm.model_name
    }

@app.post("/eval")
def evaluate():
    from evaluation.run_evaluation import run_evaluation_suite
    results = run_evaluation_suite()
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
