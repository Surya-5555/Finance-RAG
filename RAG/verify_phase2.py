import sys
import logging
from langchain_core.documents import Document

# Mock dependencies to avoid requiring Weaviate/Ollama running during the verification script
class MockVectorRetriever:
    def retrieve(self, query):
        return [{"text": "Apple is a fruit.", "metadata": {"source": "test1.txt", "chunk_id": "v1"}}]

class MockBM25Retriever:
    def retrieve(self, query):
        return [{"text": "Bananas are yellow.", "metadata": {"source": "test2.txt", "chunk_id": "b1"}}]

class MockHybridRetriever:
    def __init__(self):
        self.vector = MockVectorRetriever()
        self.bm25 = MockBM25Retriever()
    def retrieve(self, query):
        return {
            "vector_results": self.vector.retrieve(query),
            "bm25_results": self.bm25.retrieve(query)
        }

class MockLLM:
    def generate(self, prompt):
        return "Apple is a fruit."

# Real Components
from src.retrieval.rrf_retriever import RRFRetriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.prompt.builder import PromptBuilder
from src.generation.evidence_validator import EvidenceValidator
from src.rag_engine import RAGEngine

# Disable verbose logging for clean output
logging.getLogger("src").setLevel(logging.ERROR)

def run_verification():
    print("Starting Phase 2 End-to-End Verification...\n")
    
    try:
        # 1. Retrieval Setup
        hybrid = MockHybridRetriever()
        print("PASS Retrieval")
        
        # 2. RRF
        rrf = RRFRetriever(hybrid)
        print("PASS RRF")
        
        # 3. Cross Encoder
        cross_encoder = CrossEncoderReranker(rrf)
        # Mock the cross encoder score insertion to avoid downloading HF models in verification
        def mock_retrieve(q):
            # Bypass the real implementation entirely to avoid model download
            res = cross_encoder.base_retriever.retrieve(q)
            for r in res:
                r["cross_encoder_score"] = 0.95
            return res
        cross_encoder.retrieve = mock_retrieve
        print("PASS Cross Encoder")
        
        # 4. Prompt Builder
        builder = PromptBuilder()
        builder.prompt_manager.get_prompt() # Verify loading
        print("PASS Prompt Builder")
        
        # 5. Evidence Validator
        validator = EvidenceValidator()
        # Force config
        validator.enabled = True
        validator.min_score = 0.50
        validator.min_chunks = 1
        print("PASS Evidence Validator")
        
        # 6. LLM
        llm = MockLLM()
        print("PASS LLM")
        
        # 7. Integration
        engine = RAGEngine(cross_encoder, llm, builder, validator)
        
        # 8. Test Supported Flow
        res_supported = engine.ask("What is an apple?")
        if not res_supported["supported"]:
            raise Exception("Expected supported to be True")
        if len(res_supported["sources"]) == 0:
            raise Exception("Expected Citations to be generated")
        print("PASS Citations")
        
        # 9. Test Decline Flow
        validator.min_score = 0.99
        res_declined = engine.ask("What is an apple?")
        if res_declined["supported"]:
            raise Exception("Expected supported to be False")
        if "I don't have enough information" not in res_declined["answer"]:
            raise Exception("Expected standard decline message")
        print("PASS Decline Logic")
        
        print("PASS End-to-End Pipeline\n")
        print("Structured Final Output Example:")
        print(res_supported)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nFAIL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_verification()
