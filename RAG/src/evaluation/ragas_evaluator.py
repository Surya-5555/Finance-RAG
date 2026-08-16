"""
RAGAS integration for automated retrieval and generation evaluation.
"""
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
except ImportError:
    evaluate = None
    faithfulness = None
    answer_relevancy = None
    context_precision = None
    context_recall = None

logger = logging.getLogger(__name__)

class RagasEvaluator:
    """
    Wrapper for RAGAS evaluation framework.
    """

    def __init__(self):
        """
        Initialize the evaluator by mapping configured metrics.
        """
        self._load_config()
        
        # Map string names from config to actual Ragas metric objects
        self.metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall
        }
        
    def _load_config(self) -> None:
        """Load evaluation config from config.yaml."""
        self.enabled = False
        self.configured_metrics = []
        
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                eval_config = config.get("evaluation", {})
                self.enabled = eval_config.get("enabled", False)
                self.configured_metrics = eval_config.get("metrics", [])
                
    def get_active_metrics(self) -> List[Any]:
        """
        Return the list of active RAGAS metric objects based on config.
        """
        active = []
        for name in self.configured_metrics:
            if name in self.metric_map:
                active.append(self.metric_map[name])
            else:
                logger.warning(f"Unknown metric configured: {name}")
        return active

    def evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate a batch of RAG pipeline results using RAGAS.
        
        Args:
            results (List[Dict[str, Any]]): A list of dictionaries containing:
                - question (str)
                - answer (str)
                - contexts (List[str])
                - ground_truth (str)
                
        Returns:
            Dict[str, float]: A dictionary of aggregated metric scores.
        """
        if not self.enabled:
            logger.info("Evaluation is disabled in config.")
            return {}
            
        if evaluate is None:
            logger.error("RAGAS is not installed. Please run `pip install ragas datasets`.")
            return {}

        active_metrics = self.get_active_metrics()
        if not active_metrics:
            logger.warning("No valid metrics configured for evaluation.")
            return {}

        # Convert our list of dicts to the column-based format required by HuggingFace Datasets
        data = {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results]
        }
        
        dataset = Dataset.from_dict(data)
        
        logger.info(f"Running RAGAS evaluation on {len(results)} samples using metrics: {self.configured_metrics}")
        
        # Execute RAGAS
        # Note: RAGAS will automatically use the OpenAI API key in the environment for evaluation by default,
        # or fall back to whatever default LLM LangChain has configured in the environment.
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=active_metrics
        )
        
        # Convert RagasResult to standard dictionary
        return dict(evaluation_result)
