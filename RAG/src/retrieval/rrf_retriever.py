"""
RRF Retriever wrapper for the RAG pipeline.
"""
import time
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List
from src.retrieval.base import BaseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

class RRFRetriever(BaseRetriever):
    """
    Executes hybrid retrieval and fuses the results using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, hybrid_retriever: HybridRetriever) -> None:
        """
        Initialize the RRFRetriever.

        Args:
            hybrid_retriever (HybridRetriever): The configured HybridRetriever instance.
        """
        self.hybrid_retriever = hybrid_retriever
        self._load_config()

    def _load_config(self) -> None:
        """
        Load RRF configuration from config.yaml.
        """
        self.rrf_enabled = True
        self.rrf_k = 60
        
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                
            rrf_config = config.get("rrf", {})
            self.rrf_enabled = rrf_config.get("enabled", True)
            self.rrf_k = rrf_config.get("k", 60)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search and fuse with RRF.

        Args:
            query (str): The search query string.

        Returns:
            List[Dict[str, Any]]: A list of fused and ranked results.
            
        Raises:
            ValueError: If the query is empty or not a string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        # Execute hybrid retrieval
        hybrid_results = self.hybrid_retriever.retrieve(query)
        
        vector_count = len(hybrid_results.get("vector_results", []))
        bm25_count = len(hybrid_results.get("bm25_results", []))
        
        if not self.rrf_enabled:
            logger.info("RRF is disabled. Returning empty list.")
            return []
            
        start_time = time.time()
        
        # Apply RRF Fusion
        fused_results = reciprocal_rank_fusion(
            retrieval_results=hybrid_results,
            k=self.rrf_k
        )
        
        execution_time = time.time() - start_time
        final_count = len(fused_results)
        duplicates_merged = (vector_count + bm25_count) - final_count
        
        # Log structured metrics
        logger.info(
            f"RRF Fusion completed in {execution_time:.3f}s (k={self.rrf_k}). "
            f"Inputs: BM25={bm25_count}, Vector={vector_count}. "
            f"Merged {duplicates_merged} duplicates. Final count: {final_count}."
        )

        return fused_results
