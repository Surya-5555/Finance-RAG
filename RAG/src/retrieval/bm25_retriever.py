"""
BM25 retrieval implementation for the RAG pipeline.
"""
import yaml
import time
import logging
from pathlib import Path
from typing import Any
from src.vectorstore.weaviate_store import WeaviateVectorStore
from src.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)

class BM25Retriever(BaseRetriever):
    """
    Orchestrates the retrieval of documents using BM25 keyword search.
    """

    def __init__(self, vector_store: WeaviateVectorStore) -> None:
        """
        Initialize the BM25Retriever.

        Args:
            vector_store (WeaviateVectorStore): The vector store instance which supports keyword search.
        """
        self.vector_store = vector_store
        self.bm25_top_k = self._load_bm25_top_k()

    def _load_bm25_top_k(self) -> int:
        """
        Load the BM25 retrieval limit (bm25_top_k) from the config.yaml file.
        
        Returns:
            int: The maximum number of documents to retrieve.
        """
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if not config_path.exists():
            return 10
            
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            
        bm25_top_k = config.get("bm25_top_k", 10)
        
        if not isinstance(bm25_top_k, int) or bm25_top_k <= 0:
            raise ValueError("bm25_top_k in config must be a positive integer.")
            
        return bm25_top_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a given query using BM25 keyword search.

        Args:
            query (str): The search query string.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing retrieved chunks, metadata, and BM25 scores.
            
        Raises:
            ValueError: If the query is empty or not a string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        start_time = time.time()

        # Perform BM25 keyword search
        results = self.vector_store.keyword_search(query=query, limit=self.bm25_top_k)
        
        execution_time = time.time() - start_time
        logger.info(f"BM25 retrieval completed in {execution_time:.3f}s. Retrieved {len(results)} chunks.")

        # Return retrieved chunks in normalized format
        return results
