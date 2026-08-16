"""
Base retrieval interface for the RAG pipeline.
"""
from abc import ABC, abstractmethod
from typing import Any

class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval implementations.
    """

    @abstractmethod
    def retrieve(self, query: str) -> Any:
        """
        Retrieve relevant contexts for a given query.

        Args:
            query (str): The search query string.

        Returns:
            Any: The retrieved contexts. Return type depends on the specific retriever implementation.
                 Standard retrievers return list[dict[str, Any]]. 
                 Composite/Hybrid retrievers may return dict[str, list[dict[str, Any]]].
        """
        pass
