import logging
from typing import Any, List, Dict
from src.retrieval.base import BaseRetriever
from src.retrieval.retriever import VectorRetriever
from src.retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """
    Composite retriever that executes both vector search and BM25 search,
    and merges the results using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever, top_k: int = 15) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        vector_results = self.vector_retriever.retrieve(query)
        bm25_results = self.bm25_retriever.retrieve(query)

        # Merge using Reciprocal Rank Fusion
        fused_scores = {}
        doc_map = {}
        
        k = 60 # RRF constant

        for rank, doc in enumerate(vector_results):
            chunk_id = doc["document"].metadata.get("chunk_id", str(hash(doc["document"].page_content)))
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
                doc_map[chunk_id] = doc["document"]
            fused_scores[chunk_id] += 1.0 / (rank + k)
            
        for rank, doc in enumerate(bm25_results):
            chunk_id = doc["document"].metadata.get("chunk_id", str(hash(doc["document"].page_content)))
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
                doc_map[chunk_id] = doc["document"]
            fused_scores[chunk_id] += 1.0 / (rank + k)

        # Sort by fused score
        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        merged_results = []
        for chunk_id, score in sorted_chunks[:self.top_k]:
            merged_results.append({
                "document": doc_map[chunk_id],
                "rrf_score": score
            })

        logger.info(f"Hybrid retrieval merged {len(merged_results)} unique chunks.")
        return merged_results
