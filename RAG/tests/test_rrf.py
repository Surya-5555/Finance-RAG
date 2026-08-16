import pytest
from langchain_core.documents import Document
from src.retrieval.rrf import reciprocal_rank_fusion

def test_rrf_calculation_and_sorting():
    # Setup mock hybrid results
    # Vector found chunk A at rank 1, chunk B at rank 2
    # BM25 found chunk B at rank 1, chunk C at rank 2
    hybrid_results = {
        "vector": [
            {"text": "Chunk A", "metadata": {"chunk_id": "A"}},
            {"text": "Chunk B", "metadata": {"chunk_id": "B"}},
        ],
        "bm25": [
            {"text": "Chunk B", "metadata": {"chunk_id": "B"}},
            {"text": "Chunk C", "metadata": {"chunk_id": "C"}},
        ]
    }
    
    k = 60
    fused = reciprocal_rank_fusion(hybrid_results, k=k)
    
    # Expected scores:
    # A (Vector rank 1) = 1/(60+1) = 1/61 ≈ 0.01639
    # B (Vector rank 2, BM25 rank 1) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.01612 + 0.01639 = 0.0325
    # C (BM25 rank 2) = 1/(60+2) = 1/62 ≈ 0.01612
    # Order should be B, A, C
    
    assert len(fused) == 3
    
    assert fused[0]["document"].metadata["chunk_id"] == "B"
    assert "vector" in fused[0]["retrievers"]
    assert "bm25" in fused[0]["retrievers"]
    assert fused[0]["rrf_score"] == (1/(k+1) + 1/(k+2))
    
    assert fused[1]["document"].metadata["chunk_id"] == "A"
    assert fused[1]["retrievers"] == ["vector"]
    assert fused[1]["rrf_score"] == 1/(k+1)
    
    assert fused[2]["document"].metadata["chunk_id"] == "C"
    assert fused[2]["retrievers"] == ["bm25"]
    assert fused[2]["rrf_score"] == 1/(k+2)

def test_rrf_metadata_and_document_preservation():
    hybrid_results = {
        "vector": [
            {"text": "Test text", "metadata": {"chunk_id": "X", "source": "test.pdf", "page": 42}},
        ]
    }
    
    fused = reciprocal_rank_fusion(hybrid_results)
    assert len(fused) == 1
    
    item = fused[0]
    # Check LangChain Document
    assert isinstance(item["document"], Document)
    assert item["document"].page_content == "Test text"
    
    # Check Metadata
    assert item["document"].metadata["chunk_id"] == "X"
    assert item["document"].metadata["source"] == "test.pdf"
    assert item["document"].metadata["page"] == 42

def test_rrf_fallback_to_text_deduplication():
    # If chunk_id is missing, it should deduplicate based on text
    hybrid_results = {
        "vector": [
            {"text": "Identical Text", "metadata": {"source": "doc1"}},
        ],
        "bm25": [
            {"text": "Identical Text", "metadata": {"source": "doc1"}},
        ]
    }
    
    fused = reciprocal_rank_fusion(hybrid_results)
    assert len(fused) == 1  # Deduplicated!
    assert "vector" in fused[0]["retrievers"]
    assert "bm25" in fused[0]["retrievers"]

def test_rrf_empty_lists():
    hybrid_results = {
        "vector": [],
        "bm25": []
    }
    fused = reciprocal_rank_fusion(hybrid_results)
    assert len(fused) == 0

def test_rrf_different_k():
    hybrid_results = {
        "vector": [
            {"text": "Chunk A", "metadata": {"chunk_id": "A"}},
        ]
    }
    
    fused_k60 = reciprocal_rank_fusion(hybrid_results, k=60)
    assert fused_k60[0]["rrf_score"] == 1/61
    
    fused_k10 = reciprocal_rank_fusion(hybrid_results, k=10)
    assert fused_k10[0]["rrf_score"] == 1/11
