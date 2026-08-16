import pytest
from unittest.mock import MagicMock
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import VectorRetriever

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    
    # Mock similarity_search
    store.similarity_search.return_value = [
        {"text": "Vector match 1", "metadata": {"source": "doc1.txt", "chunk_id": "1"}, "distance": 0.1},
        {"text": "Vector match 2", "metadata": {"source": "doc2.txt", "chunk_id": "2"}, "distance": 0.2}
    ]
    
    # Mock keyword_search
    store.keyword_search.return_value = [
        {"text": "BM25 match 1", "metadata": {"source": "doc3.txt", "chunk_id": "3"}, "score": 2.5},
        {"text": "BM25 match 2", "metadata": {"source": "doc4.txt", "chunk_id": "4"}, "score": 1.2}
    ]
    
    return store

@pytest.fixture
def mock_embedding_provider():
    provider = MagicMock()
    provider.embed_query.return_value = [0.1, 0.2, 0.3]
    return provider

@pytest.fixture
def bm25_retriever(mock_vector_store, monkeypatch):
    monkeypatch.setattr(BM25Retriever, "_load_bm25_top_k", lambda self: 2)
    return BM25Retriever(vector_store=mock_vector_store)

@pytest.fixture
def vector_retriever(mock_embedding_provider, mock_vector_store, monkeypatch):
    monkeypatch.setattr(VectorRetriever, "_load_vector_top_k", lambda self: 2)
    return VectorRetriever(embedding_provider=mock_embedding_provider, vector_store=mock_vector_store)

@pytest.fixture
def hybrid_retriever(vector_retriever, bm25_retriever):
    return HybridRetriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)

def test_bm25_retriever(bm25_retriever, mock_vector_store):
    query = "keyword test"
    results = bm25_retriever.retrieve(query)
    
    mock_vector_store.keyword_search.assert_called_once_with(query=query, limit=2)
    
    assert len(results) == 2
    assert results[0]["text"] == "BM25 match 1"
    assert "score" in results[0]
    assert results[0]["metadata"]["source"] == "doc3.txt"
    assert "distance" not in results[0]

def test_hybrid_retriever(hybrid_retriever, mock_vector_store, mock_embedding_provider):
    query = "hybrid test"
    results = hybrid_retriever.retrieve(query)
    
    mock_embedding_provider.embed_query.assert_called_once_with(query)
    mock_vector_store.similarity_search.assert_called_once()
    mock_vector_store.keyword_search.assert_called_once_with(query=query, limit=2)
    
    assert isinstance(results, dict)
    assert "vector_results" in results
    assert "bm25_results" in results
    
    vector_res = results["vector_results"]
    bm25_res = results["bm25_results"]
    
    assert len(vector_res) == 2
    assert len(bm25_res) == 2
    
    assert vector_res[0]["text"] == "Vector match 1"
    assert "distance" in vector_res[0]
    
    assert bm25_res[0]["text"] == "BM25 match 1"
    assert "score" in bm25_res[0]
    
def test_empty_query_raises_error(hybrid_retriever):
    with pytest.raises(ValueError):
        hybrid_retriever.retrieve("")
        
    with pytest.raises(ValueError):
        hybrid_retriever.retrieve("   ")
