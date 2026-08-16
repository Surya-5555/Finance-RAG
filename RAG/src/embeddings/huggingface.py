"""
HuggingFace implementation of the embedding provider.
"""
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .base import BaseEmbeddingProvider


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider using local HuggingFace models via sentence-transformers.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the HuggingFace embedding provider.

        Args:
            model_name (str | None): The name of the model to use. If None, it will be loaded from config.yaml.
        """
        if model_name is None:
            model_name = self._load_model_name_from_config()
            
        if not model_name or not isinstance(model_name, str):
            raise ValueError("A valid embedding model name must be provided in config or as an argument.")

        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name)

    def _load_model_name_from_config(self) -> str:
        """
        Load the embedding model name from the config.yaml file.
        
        Returns:
            str: The embedding model name.
            
        Raises:
            FileNotFoundError: If the config file cannot be found.
            KeyError: If embedding_model is not defined in the config.
        """
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            
        model_name = config.get("embedding_model")
        if not model_name:
            raise KeyError("'embedding_model' key is missing from config.yaml")
            
        return model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text documents.

        Args:
            texts (list[str]): A list of strings to embed.

        Returns:
            list[list[float]]: A list of embedding vectors.
            
        Raises:
            ValueError: If texts is not a list of strings.
        """
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input texts must be a list of strings.")
            
        if not texts:
            return []

        # encode() returns a numpy array or tensor, so we convert it to a list of lists of floats
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            text (str): The query string to embed.

        Returns:
            list[float]: The embedding vector for the query.
            
        Raises:
            ValueError: If text is not a string or is empty.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        # encode() returns a numpy array, convert to a list of floats
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        return embedding
