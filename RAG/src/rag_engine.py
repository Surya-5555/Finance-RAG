import os
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RAGEngine:
    """End-to-end RAG pipeline orchestrator."""
    
    def __init__(self, retriever, llm, prompt_builder, evidence_validator):
        """
        Initialize the RAGEngine.
        
        Args:
            retriever: The retrieval pipeline (e.g. CrossEncoderReranker).
            llm: The generation model interface.
            prompt_builder: The PromptBuilder instance.
            evidence_validator: The EvidenceValidator instance.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.evidence_validator = evidence_validator
        
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Execute the RAG pipeline for a given query.
        
        Returns:
            Dict containing the answer, support boolean, and citation sources.
        """
        total_start = time.time()
        
        # 1. Retrieve contexts
        retrieval_start = time.time()
        contexts = self.retriever.retrieve(query)
        retrieval_latency = time.time() - retrieval_start
        
        # 2. Validate Evidence
        val_start = time.time()
        validation = self.evidence_validator.validate(contexts)
        val_latency = time.time() - val_start
        
        if not validation.get("supported", False):
            total_latency = time.time() - total_start
            logger.info(f"Query declined. Total Latency: {total_latency:.3f}s")
            return {
                "answer": self.evidence_validator.get_decline_response(),
                "supported": False,
                "sources": []
            }
            
        # 3. Build Prompt
        prompt_start = time.time()
        prompt = self.prompt_builder.build(query, contexts)
        prompt_latency = time.time() - prompt_start
        
        # 4. Generate Answer
        llm_start = time.time()
        response = self.llm.generate(prompt)
        llm_latency = time.time() - llm_start
        
        # 5. Extract Citations
        sources_list = []
        seen_chunks = set()
        
        for doc_item in contexts:
            doc = doc_item.get("document")
            if not doc:
                continue
                
            meta = doc.metadata
            chunk_id = meta.get("chunk_id")
            
            # Deduplicate strictly by chunk_id if available, otherwise by page_content
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
            "answer": response.strip(),
            "supported": True,
            "sources": sources_list
        }