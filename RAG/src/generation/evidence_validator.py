"""
Evidence Validator for strictly enforcing citation quality and preventing hallucinations.
"""
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class EvidenceValidator:
    """
    Validates the retrieved context against configured thresholds before allowing LLM generation.
    """

    def __init__(self) -> None:
        """
        Initialize the EvidenceValidator.
        """
        self._load_config()

    def _load_config(self) -> None:
        """
        Load citation thresholds from config.yaml.
        """
        self.enabled = True
        self.min_score = 0.50
        self.min_chunks = 2
        
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                
            citation_config = config.get("citation", {})
            self.enabled = citation_config.get("enabled", True)
            self.min_score = citation_config.get("minimum_cross_encoder_score", 0.50)
            self.min_chunks = citation_config.get("minimum_chunks", 2)

    def validate(self, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the retrieved chunks against thresholds.
        
        Args:
            contexts (List[Dict[str, Any]]): The retrieved Top-N chunks.
            
        Returns:
            Dict[str, Any]: Validation result. 
                            If successful: {"supported": True}
                            If failed: {"supported": False, "reason": "..."}
        """
        if not self.enabled:
            logger.info("Evidence validation is disabled. Bypassing checks.")
            return {"supported": True}
            
        total_chunks = len(contexts)
        
        if total_chunks == 0:
            logger.warning("Evidence validation failed: No chunks retrieved.")
            return {
                "supported": False,
                "reason": "Insufficient supporting evidence. No documents retrieved."
            }
            
        valid_chunks_count = 0
        
        for chunk in contexts:
            score = chunk.get("cross_encoder_score", 0.0)
            if score >= self.min_score:
                valid_chunks_count += 1
                
        if valid_chunks_count < self.min_chunks:
            logger.warning(
                f"Evidence validation failed: Only {valid_chunks_count}/{total_chunks} chunks "
                f"met the threshold of {self.min_score} (required: {self.min_chunks}). LLM invocation skipped."
            )
            return {
                "supported": False,
                "reason": f"Insufficient supporting evidence. Only {valid_chunks_count} highly relevant sources found."
            }
            
        logger.info(f"Evidence validation passed: {valid_chunks_count} chunks met the score threshold of {self.min_score}.")
        return {"supported": True}

    def get_decline_response(self) -> str:
        """
        Returns the standard, non-hallucinated response for unsupported questions.
        """
        return "I don't have enough information in the retrieved documents to answer this question reliably."
