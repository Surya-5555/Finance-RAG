import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker

@pytest.fixture
def mock_base_retriever():
    retriever = MagicMock()
    # Mock RRF output
    retriever.retrieve.return_value = [
        {
            "document": Document(page_content="Apple is a fruit", metadata={"id": 1}),
            "rrf_score": 0.05,
            "retrievers": ["bm25"]
        },
        {
            "document": Document(page_content="Apple is a tech company", metadata={"id": 2}),
            "rrf_score": 0.03,
            "retrievers": ["vector"]
        },
        {
            "document": Document(page_content="Bananas are yellow", metadata={"id": 3}),
            "rrf_score": 0.04,
            "retrievers": ["bm25", "vector"]
        }
    ]
    return retriever

@pytest.fixture
def mock_cross_encoder():
    model = MagicMock()
    # Predict returns scores corresponding to the input pairs
    # Pair 1: "Apple" + "Apple is a fruit" -> Score: 0.1
    # Pair 2: "Apple" + "Apple is a tech company" -> Score: 0.9 (highest)
    # Pair 3: "Apple" + "Bananas are yellow" -> Score: -2.0
    model.predict.return_value = [0.1, 0.9, -2.0]
    return model

@pytest.fixture
def reranker(mock_base_retriever, mock_cross_encoder, monkeypatch):
    reranker_instance = CrossEncoderReranker(base_retriever=mock_base_retriever)
    
    # Mock config to avoid reading real file
    reranker_instance.enabled = True
    reranker_instance.top_n = 2
    reranker_instance.model_name = "test-model"
    
    # Inject mock model to avoid downloading
    reranker_instance._model = mock_cross_encoder
    
    return reranker_instance

def test_reranker_sorting_and_top_n(reranker, mock_cross_encoder):
    query = "Apple computer"
    results = reranker.retrieve(query)
    
    mock_cross_encoder.predict.assert_called_once()
    
    # Check Top-N filtering (top_n = 2)
    assert len(results) == 2
    
    # Check sorting (highest score first)
    assert results[0]["document"].metadata["id"] == 2
    assert results[0]["cross_encoder_score"] == 0.9
    
    assert results[1]["document"].metadata["id"] == 1
    assert results[1]["cross_encoder_score"] == 0.1

def test_reranker_preservation(reranker):
    query = "Apple computer"
    results = reranker.retrieve(query)
    
    # Metadata and Document preservation
    assert isinstance(results[0]["document"], Document)
    assert results[0]["document"].page_content == "Apple is a tech company"
    
    # RRF Score and Provenance preservation
    assert results[0]["rrf_score"] == 0.03
    assert results[0]["retrievers"] == ["vector"]

def test_empty_query_raises_error(reranker):
    with pytest.raises(ValueError):
        reranker.retrieve("")
        
def test_empty_results_handling(reranker, mock_base_retriever, mock_cross_encoder):
    mock_base_retriever.retrieve.return_value = []
    
    results = reranker.retrieve("test")
    
    assert len(results) == 0
    mock_cross_encoder.predict.assert_not_called()

def test_single_document(reranker, mock_base_retriever, mock_cross_encoder):
    mock_base_retriever.retrieve.return_value = [
        {
            "document": Document(page_content="Single doc", metadata={"id": 1}),
            "rrf_score": 0.05,
            "retrievers": ["bm25"]
        }
    ]
    mock_cross_encoder.predict.return_value = [0.8]
    
    results = reranker.retrieve("test")
    
    assert len(results) == 1
    assert results[0]["cross_encoder_score"] == 0.8
    assert results[0]["document"].metadata["id"] == 1

def test_disabled_reranker(reranker, mock_cross_encoder):
    reranker.enabled = False
    
    results = reranker.retrieve("test")
    
    # Should return base results immediately, without calling predict
    mock_cross_encoder.predict.assert_not_called()
    assert len(results) == 2 # Base returns 3, but top_n is 2, so it truncates
