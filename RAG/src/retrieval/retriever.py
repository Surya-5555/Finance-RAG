import time
import logging
from typing import Any, List, Dict
from src.vectorstore.chroma_store import ChromaDBStore
from src.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)

class VectorRetriever(BaseRetriever):
    """
    Orchestrates the retrieval of documents using Semantic (Vector) search.
    """

    def __init__(self, vector_store: ChromaDBStore, top_k: int = 15):
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        start_time = time.time()

        # Semantic search
        docs = self.vector_store.search(query=query, k=self.top_k)
        
        results = []
        for doc in docs:
            results.append({
                "document": doc,
                "vector_score": 1.0 # Chroma doesn't natively expose distance in the basic langchain wrapper in this simplified version
            })

        execution_time = time.time() - start_time
        logger.info(f"Vector retrieval completed in {execution_time:.3f}s. Retrieved {len(results)} chunks.")

        return results
