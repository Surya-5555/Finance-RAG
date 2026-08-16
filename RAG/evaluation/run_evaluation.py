import os
import sys
import json
import yaml
import logging
from pathlib import Path

# Add project root to path so we can import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluation.ragas_evaluator import RagasEvaluator
from src.rag_engine import RAGEngine
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.prompt.builder import PromptBuilder
from src.prompt.prompt_manager import PromptManager
from src.generation.evidence_validator import EvidenceValidator
from src.llm.openai_provider import OpenAILLMProvider
from src.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.vectorstore.chroma_store import ChromaDBStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_record(record: dict, index: int, all_ids: set) -> bool:
    """Validate a single dataset record against the required schema."""
    required_keys = {"id", "question", "reference_answer", "expected_sources"}
    if not required_keys.issubset(record.keys()):
        logger.error(f"Row {index} missing keys. Required: {required_keys}")
        return False
        
    rec_id = record.get("id")
    if not rec_id or rec_id in all_ids:
        logger.error(f"Row {index} has duplicate or missing ID: {rec_id}")
        return False
    all_ids.add(rec_id)
    
    question = record.get("question")
    if not question or not str(question).strip():
        logger.error(f"Row {index} has empty question.")
        return False
        
    ref_ans = record.get("reference_answer")
    if not ref_ans or not str(ref_ans).strip():
        logger.error(f"Row {index} has empty reference_answer.")
        return False
    sources = record["expected_sources"]
    if not isinstance(sources, list):
        logger.error(f"Row {index} 'expected_sources' must be a list.")
        return False
        
    for s in sources:
        if not isinstance(s, dict) or "source" not in s:
            logger.error(f"Row {index} 'expected_sources' items must contain 'source' key.")
            return False
            
    return True

def load_dataset(filepath: str) -> list:
    """Load and validate the golden dataset from a JSONL file."""
    data = []
    path = Path(filepath)
    if not path.exists():
        logger.error(f"Dataset not found: {filepath}")
        return data
        
    all_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Row {i} is not valid JSON: {e}")
                sys.exit(1)
                
            if not validate_record(record, i, all_ids):
                sys.exit(1)
                
            data.append(record)
    return data

def load_thresholds() -> dict:
    """Load metric thresholds from config.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("evaluation", {}).get("threshold", {})
    return {}

def setup_real_engine():
    """Setup the real RAGEngine using production components."""
    logger.info("Initializing Real RAGEngine Pipeline components...")
    embed_provider = OpenAIEmbeddingProvider()
    vector_store = ChromaDBStore("./chroma_db", "finance_docs", embed_provider)
    
    vector = VectorRetriever(vector_store)
    bm25 = BM25Retriever(vector_store)
    hybrid = HybridRetriever(vector, bm25)
    reranker = CrossEncoderReranker(hybrid)
    
    llm = OpenAILLMProvider(temperature=0.0)
    prompt_manager = PromptManager()
    builder = PromptBuilder(prompt_manager)
    validator = EvidenceValidator()
    
    return RAGEngine(reranker, llm, builder, validator)

def run_evaluation_suite():
    dataset_path = Path(__file__).resolve().parent / "golden_dataset.jsonl"
    dataset = load_dataset(str(dataset_path))
    
    if not dataset:
        logger.error("Evaluation aborted: Dataset is empty or missing.")
        return {"error": "Dataset empty or missing"}
        
    logger.info(f"Loaded {len(dataset)} evaluation samples.")
    
    try:
        engine = setup_real_engine()
    except Exception as e:
        logger.warning(f"Failed to initialize real RAG pipeline: {e}")
        return {"error": str(e)}
        
    evaluator = RagasEvaluator()
    thresholds = load_thresholds()
    
    if not evaluator.enabled:
        logger.warning("Evaluation is disabled in config.yaml. Aborting.")
        return {"error": "Evaluation disabled"}
        
    evaluation_inputs = []
    
    supported_count = 0
    unsupported_count = 0
    successful_runs = 0
    
    for idx, record in enumerate(dataset):
        question = record["question"]
        ground_truth = record["reference_answer"]
        
        try:
            result = engine.ask(question)
            
            raw_contexts = engine.retriever.retrieve(question)
            retrieved_contexts = [item["document"].page_content for item in raw_contexts]
            
            evaluation_inputs.append({
                "question": question,
                "answer": result["answer"],
                "contexts": retrieved_contexts,
                "ground_truth": ground_truth
            })
            
            if result.get("supported"):
                supported_count += 1
            else:
                unsupported_count += 1
                
            successful_runs += 1
                
        except Exception as e:
            logger.warning(f"RAGEngine execution failed for Q{idx+1} '{question[:30]}...'. Error: {e}")
            
    if successful_runs == 0:
        logger.warning("All pipeline executions failed!")
        return {"error": "All executions failed"}
        
    try:
        metrics = evaluator.evaluate(evaluation_inputs)
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}")
        return {"error": str(e)}
        
    overall_pass = True
    for metric_name, score in metrics.items():
        threshold = thresholds.get(metric_name, 0.0)
        passed = score >= threshold
        if not passed:
            overall_pass = False
            
    metrics["overall_pass"] = overall_pass
    metrics["supported_count"] = supported_count
    metrics["unsupported_count"] = unsupported_count
    metrics["total"] = successful_runs
    
    return metrics

if __name__ == "__main__":
    result = run_evaluation_suite()
    print(json.dumps(result, indent=2))
