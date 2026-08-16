import os
import time
import logging
from typing import Dict, Any, List

# Try to import Langfuse, default to none if not available
try:
    from langfuse.decorators import observe
    from langfuse import Langfuse
    # Initialize Langfuse client
    langfuse = Langfuse()
    LANGFUSE_ENABLED = True
except ImportError:
    LANGFUSE_ENABLED = False
    
    # Create a dummy decorator if Langfuse is not available
    def observe(**kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

class RAGEngine:
    """End-to-end RAG pipeline orchestrator with Langfuse observability."""
    
    def __init__(self, retriever, llm, prompt_builder, evidence_validator):
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.evidence_validator = evidence_validator
        
    @observe(name="rag_ask")
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Execute the RAG pipeline for a given query.
        """
        total_start = time.time()
        
        # 1. Query Understanding Layer (Classification)
        # Simplified: Check if query contains "compare", "vs", "versus"
        is_comparison = any(term in query.lower() for term in ["compare", "vs", "versus", "trend"])
        
        # 2. Retrieve contexts
        retrieval_start = time.time()
        
        # If comparison, we might run multi-query in a full implementation. 
        # Here we just use the hybrid retriever which retrieves top N.
        contexts = self.retriever.retrieve(query)
        retrieval_latency = time.time() - retrieval_start
            
        # 3. Build Prompt
        prompt_start = time.time()
        prompt = self.prompt_builder.build(query, contexts)
        prompt_latency = time.time() - prompt_start
        
        # 4. Generate Answer
        llm_start = time.time()
        answer = self.llm.generate(prompt)
        llm_latency = time.time() - llm_start
        
        # 5. Citation Enforcement
        val_start = time.time()
        is_valid = self.evidence_validator.validate_answer(answer, contexts)
        val_latency = time.time() - val_start
        
        if not is_valid:
            answer = self.evidence_validator.get_decline_response()
        
        # 6. Extract Citations
        sources_list = []
        seen_chunks = set()
        
        for doc_item in contexts:
            doc = doc_item.get("document")
            if not doc:
                continue
                
            meta = doc.metadata
            chunk_id = meta.get("chunk_id")
            
            unique_key = chunk_id if chunk_id else hash(doc.page_content)
            
            if unique_key not in seen_chunks:
                seen_chunks.add(unique_key)
                source = meta.get("source", "unknown")
                sources_list.append({
                    "source": source,
                    "filename": meta.get("filename") or os.path.basename(source),
                    "page": meta.get("page"),
                    "title": meta.get("title"),
                    "chunk_id": chunk_id
                })
                
        total_latency = time.time() - total_start
        
        logger.info(
            f"Pipeline Complete | Total: {total_latency:.3f}s | "
            f"Retrieval: {retrieval_latency:.3f}s | Val: {val_latency:.3f}s | "
            f"Prompt: {prompt_latency:.3f}s | LLM: {llm_latency:.3f}s | Citations: {len(sources_list)}"
        )
        
        return {
            "answer": answer.strip(),
            "supported": is_valid,
            "sources": sources_list
        }