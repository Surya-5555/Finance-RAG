"""
Hybrid retrieval implementation for the RAG pipeline.
"""
from typing import Any
from src.retrieval.base import BaseRetriever
from src.retrieval.retriever import VectorRetriever
from src.retrieval.bm25_retriever import BM25Retriever

class HybridRetriever(BaseRetriever):
    """
    Composite retriever that executes both vector search and BM25 search.
    Currently returns both result sets separately without merging or reranking.
    """

    def __init__(self, vector_retriever: VectorRetriever, bm25_retriever: BM25Retriever) -> None:
        """
        Initialize the HybridRetriever.

        Args:
            vector_retriever (VectorRetriever): Configured vector retriever.
            bm25_retriever (BM25Retriever): Configured BM25 retriever.
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str) -> dict[str, list[dict[str, Any]]]:
        """
        Retrieve relevant documents using both vector and BM25 retrievers independently.

        Args:
            query (str): The search query string.

        Returns:
            dict[str, list[dict[str, Any]]]: A dictionary containing two keys:
                - 'vector_results': List of results from the vector retriever.
                - 'bm25_results': List of results from the BM25 retriever.
            
        Raises:
            ValueError: If the query is empty or not a string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        # Execute both retrievers independently
        vector_results = self.vector_retriever.retrieve(query)
        bm25_results = self.bm25_retriever.retrieve(query)

        # Return both sets separately (no fusion yet)
        return {
            "vector_results": vector_results,
            "bm25_results": bm25_results
        }
