"""
Vector retrieval implementation for the RAG pipeline.
"""
import yaml
import time
import logging
from pathlib import Path
from typing import Any
from src.embeddings.base import BaseEmbeddingProvider
from src.vectorstore.weaviate_store import WeaviateVectorStore
from src.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)

class VectorRetriever(BaseRetriever):
    """
    Orchestrates the retrieval of documents using a vector store and embedding provider.
    """

    def __init__(self, embedding_provider: BaseEmbeddingProvider, vector_store: WeaviateVectorStore) -> None:
        """
        Initialize the VectorRetriever.

        Args:
            embedding_provider (BaseEmbeddingProvider): The embedding provider instance.
            vector_store (WeaviateVectorStore): The vector store instance.
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.vector_top_k = self._load_vector_top_k()

    def _load_vector_top_k(self) -> int:
        """
        Load the vector retrieval limit (vector_top_k) from the config.yaml file.
        
        Returns:
            int: The maximum number of documents to retrieve.
        """
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if not config_path.exists():
            return 10
            
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            
        vector_top_k = config.get("vector_top_k", 10)
        
        if not isinstance(vector_top_k, int) or vector_top_k <= 0:
            raise ValueError("vector_top_k in config must be a positive integer.")
            
        return vector_top_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a given query using vector similarity.

        Args:
            query (str): The search query string.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing retrieved chunks, metadata, and distances.
            
        Raises:
            ValueError: If the query is empty or not a string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        start_time = time.time()

        # 1. Embed the query
        query_vector = self.embedding_provider.embed_query(query)

        # 2. Perform vector similarity search
        results = self.vector_store.similarity_search(query_vector=query_vector, limit=self.vector_top_k)
        
        execution_time = time.time() - start_time
        logger.info(f"Vector retrieval completed in {execution_time:.3f}s. Retrieved {len(results)} chunks.")

        # 3. Return retrieved chunks in normalized format
        return results
