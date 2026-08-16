"""
Cross-Encoder Re-ranker module for the RAG pipeline.
"""
import time
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List
from src.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseRetriever):
    """
    Re-ranks retrieved documents using a HuggingFace Cross-Encoder model.
    Loads the model lazily upon the first request to minimize startup time.
    """

    def __init__(self, base_retriever: BaseRetriever) -> None:
        """
        Initialize the CrossEncoderReranker.

        Args:
            base_retriever (BaseRetriever): The underlying retriever (e.g., RRFRetriever) 
                                            providing the initial ranked list.
        """
        self.base_retriever = base_retriever
        self._model = None
        self._load_config()

    def _load_config(self) -> None:
        """
        Load Cross-Encoder configuration from config.yaml.
        """
        self.enabled = True
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.top_n = 5
        
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                
            ce_config = config.get("cross_encoder", {})
            self.enabled = ce_config.get("enabled", True)
            self.model_name = ce_config.get("model", self.model_name)
            self.top_n = ce_config.get("top_n", 5)

    def _get_model(self):
        """
        Lazily load the sentence-transformers CrossEncoder model.
        """
        if self._model is None:
            logger.info(f"Loading Cross-Encoder model: {self.model_name}...")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-Encoder model loaded successfully.")
        return self._model

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using the base retriever and re-rank them.

        Args:
            query (str): The search query string.

        Returns:
            List[Dict[str, Any]]: A list of Top-N re-ranked chunks.
            
        Raises:
            ValueError: If the query is empty or not a string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        # 1. Fetch initial results from the base retriever (e.g., RRF)
        results = self.base_retriever.retrieve(query)
        
        input_count = len(results)
        
        # 2. Skip re-ranking if disabled or no results
        if not self.enabled or input_count == 0:
            logger.info("Cross-Encoder is disabled or no input results. Returning base results.")
            return results[:self.top_n]
            
        start_time = time.time()
        model = self._get_model()
        
        # 3. Construct query-document pairs for the Cross-Encoder
        # The cross-encoder expects a list of [query, text] pairs
        pairs = [[query, item["document"].page_content] for item in results]
        
        # 4. Predict scores
        # We assume output is a 1D numpy array of scores
        scores = model.predict(pairs)
        
        # 5. Inject scores back into the result dictionaries
        for i, item in enumerate(results):
            item["cross_encoder_score"] = float(scores[i])
            
        # 6. Sort by cross_encoder_score descending
        results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        
        # 7. Truncate to Top-N
        top_n_results = results[:self.top_n]
        
        execution_time = time.time() - start_time
        output_count = len(top_n_results)
        
        # Calculate stats for logging
        if output_count > 0:
            top_score = top_n_results[0]["cross_encoder_score"]
            avg_score = sum(item["cross_encoder_score"] for item in top_n_results) / output_count
        else:
            top_score = 0.0
            avg_score = 0.0
            
        logger.info(
            f"Cross-Encoder Re-ranking completed in {execution_time:.3f}s. "
            f"Model: {self.model_name} | Inputs: {input_count} | Outputs: {output_count} | "
            f"Top Score: {top_score:.4f} | Avg Score: {avg_score:.4f}"
        )

        return top_n_results
