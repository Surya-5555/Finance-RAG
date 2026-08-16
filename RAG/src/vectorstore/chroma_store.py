import os
import chromadb
from typing import List, Dict, Any
from chromadb.config import Settings
from langchain_core.documents import Document

class ChromaDBStore:
    """
    Vector store implementation using ChromaDB for local persistence.
    """
    
    def __init__(self, persist_directory: str, collection_name: str, embedding_function):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize chroma client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # We need a chromadb embedding function wrapper
        from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
        class LangchainEmbeddingWrapper(EmbeddingFunction):
            def __init__(self, langchain_embeddings):
                self.langchain_embeddings = langchain_embeddings
            
            def __call__(self, input: Documents) -> Embeddings:
                return self.langchain_embeddings.embed_documents(input)
                
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=LangchainEmbeddingWrapper(self.embedding_function)
        )
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to ChromaDB.
        Expects documents as dicts with 'text' and 'metadata'.
        """
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            meta = doc["metadata"].copy()
            # Chroma only accepts str, int, float, bool in metadata
            cleaned_meta = {}
            for k, v in meta.items():
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    cleaned_meta[k] = v
                else:
                    cleaned_meta[k] = str(v)
                    
            chunk_id = cleaned_meta.get("chunk_id", str(hash(doc["text"])))
            
            ids.append(chunk_id)
            texts.append(doc["text"])
            metadatas.append(cleaned_meta)
            
        # Batch insert to avoid huge payload issues
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            
    def search(self, query: str, k: int = 5, filter_dict: dict = None) -> List[Document]:
        """
        Semantic search using ChromaDB.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter_dict
        )
        
        docs = []
        if not results['documents'] or not results['documents'][0]:
            return docs
            
        for i in range(len(results['documents'][0])):
            text = results['documents'][0][i]
            meta = results['metadatas'][0][i] if results['metadatas'] else {}
            docs.append(Document(page_content=text, metadata=meta))
            
        return docs
