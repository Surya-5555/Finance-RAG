"""
Reciprocal Rank Fusion (RRF) implementation.
"""
from typing import Any, Dict, List
from langchain_core.documents import Document

def reciprocal_rank_fusion(retrieval_results: Dict[str, List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Fuses multiple ranked retrieval lists into a single ranked list using Reciprocal Rank Fusion (RRF).
    
    RRF Score = Σ (1 / (k + rank))
    where rank starts from 1.
    
    Args:
        retrieval_results (Dict[str, List[Dict[str, Any]]]): A dictionary where keys are retriever names
            (e.g., "bm25", "vector") and values are lists of retrieved chunk dictionaries.
        k (int, optional): The RRF k constant. Defaults to 60.
        
    Returns:
        List[Dict[str, Any]]: A sorted list of fused chunks, each containing:
            - 'document': A LangChain Document object containing text and metadata.
            - 'rrf_score': The computed float RRF score.
            - 'retrievers': A list of strings indicating which retrievers found the chunk.
    """
    fused_scores: Dict[str, float] = {}
    fused_retrievers: Dict[str, List[str]] = {}
    fused_docs: Dict[str, Dict[str, Any]] = {}
    
    # Process each retriever's list
    for retriever_name, results in retrieval_results.items():
        for rank, item in enumerate(results, start=1):
            # Extract chunk_id or fallback to text for deduplication
            meta = item.get("metadata", {})
            chunk_id = meta.get("chunk_id")
            
            # If chunk_id is missing or empty, use text as fallback key
            if not chunk_id:
                chunk_id = item.get("text", "")
                
            # If we still have no valid key, skip
            if not chunk_id:
                continue
                
            # Ensure string key
            chunk_id = str(chunk_id)
            
            # Initialize if this is the first time we see this chunk
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
                fused_retrievers[chunk_id] = []
                fused_docs[chunk_id] = {
                    "text": item.get("text", ""),
                    "metadata": meta
                }
            
            # Update score
            fused_scores[chunk_id] += 1.0 / (k + rank)
            
            # Update provenance (avoid duplicates in case of weird inputs)
            if retriever_name not in fused_retrievers[chunk_id]:
                fused_retrievers[chunk_id].append(retriever_name)
                
    # Format the final output list
    final_results = []
    for chunk_id, score in fused_scores.items():
        doc_data = fused_docs[chunk_id]
        
        # Create LangChain Document
        lc_doc = Document(
            page_content=doc_data["text"],
            metadata=doc_data["metadata"]
        )
        
        final_results.append({
            "document": lc_doc,
            "rrf_score": score,
            "retrievers": fused_retrievers[chunk_id]
        })
        
    # Sort descending by RRF score
    final_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    return final_results
