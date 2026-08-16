import os
from langchain_openai import OpenAIEmbeddings

class OpenAIEmbeddingProvider:
    """
    OpenAI embedding provider for the RAG pipeline.
    Uses text-embedding-3-small by default as per requirements.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
            
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
        self.model_name = model_name
        
    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)
