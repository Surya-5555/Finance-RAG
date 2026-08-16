"""
Abstract base class for embedding providers.
"""
from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """
    Abstract interface for generating embeddings from text.
    Provides a unified contract for different embedding backends (e.g., HuggingFace, OpenAI).
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text documents.

        Args:
            texts (list[str]): A list of strings to embed.

        Returns:
            list[list[float]]: A list of embedding vectors.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            text (str): The query string to embed.

        Returns:
            list[float]: The embedding vector for the query.
        """
        pass
