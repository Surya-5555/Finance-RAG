import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import VectorRetriever

@pytest.fixture
def mock_chroma_store():
    store = MagicMock()
    # Mock search
    doc1 = Document(page_content="Vector match 1", metadata={"source": "doc1.txt", "chunk_id": "1"})
    doc2 = Document(page_content="Vector match 2", metadata={"source": "doc2.txt", "chunk_id": "2"})
    store.search.return_value = [doc1, doc2]
    
    # Mock collection.get for BM25
    store.collection.get.return_value = {
        "documents": ["BM25 match 1", "BM25 match 2"],
        "metadatas": [{"source": "doc3.txt", "chunk_id": "3"}, {"source": "doc4.txt", "chunk_id": "4"}]
    }
    return store

@pytest.fixture
def bm25_retriever(mock_chroma_store):
    return BM25Retriever(vector_store=mock_chroma_store)

@pytest.fixture
def vector_retriever(mock_chroma_store):
    return VectorRetriever(vector_store=mock_chroma_store)

@pytest.fixture
def hybrid_retriever(vector_retriever, bm25_retriever):
    return HybridRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)

def test_bm25_retriever(bm25_retriever):
    query = "keyword test"
    results = bm25_retriever.retrieve(query)
    
    assert len(results) <= 2
    
def test_vector_retriever(vector_retriever, mock_chroma_store):
    query = "vector test"
    results = vector_retriever.retrieve(query)
    mock_chroma_store.search.assert_called_once_with(query=query, k=15)
    assert len(results) == 2

def test_hybrid_retriever(hybrid_retriever):
    query = "hybrid test"
    results = hybrid_retriever.retrieve(query)
    assert isinstance(results, list)
