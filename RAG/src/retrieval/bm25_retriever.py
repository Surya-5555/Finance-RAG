import time
import logging
from typing import Any, List, Dict
from rank_bm25 import BM25Okapi
from src.vectorstore.chroma_store import ChromaDBStore

logger = logging.getLogger(__name__)

class BM25Retriever:
    """
    BM25 keyword search retriever.
    Builds the BM25 index in-memory from the persisted ChromaDB store.
    """

    def __init__(self, vector_store: ChromaDBStore, top_k: int = 10):
        self.vector_store = vector_store
        self.top_k = top_k
        self.bm25 = None
        self.documents = []
        self._build_index()

    def _build_index(self):
        """Fetch all documents from Chroma and build BM25 index."""
        logger.info("Building BM25 index from ChromaDB...")
        results = self.vector_store.collection.get()
        if not results or not results['documents']:
            logger.info("No documents found in ChromaDB for BM25.")
            return

        self.documents = []
        tokenized_corpus = []
        for i in range(len(results['documents'])):
            text = results['documents'][i]
            meta = results['metadatas'][i]
            
            self.documents.append({
                "document": type("Doc", (), {"page_content": text, "metadata": meta})(),
            })
            tokenized_corpus.append(text.lower().split())

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(self.documents)} documents.")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using BM25 keyword search.
        """
        if not self.bm25:
            # Try to build again if it was empty
            self._build_index()
            if not self.bm25:
                return []

        start_time = time.time()
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0: # Only return if there's some overlap
                doc_dict = self.documents[idx].copy()
                doc_dict["bm25_score"] = float(scores[idx])
                results.append(doc_dict)

        execution_time = time.time() - start_time
        logger.info(f"BM25 retrieval completed in {execution_time:.3f}s. Retrieved {len(results)} chunks.")

        return results
