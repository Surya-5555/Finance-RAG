import pytest
from unittest.mock import Mock
from langchain_core.documents import Document
from src.rag_engine import RAGEngine

class MockRetriever:
    def __init__(self, contexts):
        self.contexts = contexts
    def retrieve(self, query):
        return self.contexts

class MockLLM:
    def __init__(self, response):
        self.response = response
    def generate(self, prompt):
        return self.response

class MockPromptBuilder:
    def build(self, query, contexts):
        return "mock_prompt"

class MockEvidenceValidator:
    def __init__(self, supported):
        self._supported = supported
    def validate_answer(self, answer, contexts):
        return self._supported
    def get_decline_response(self):
        return "Declined."

def test_rag_engine_supported_flow():
    # Setup mocks
    contexts = [
        {
            "document": Document(page_content="Text", metadata={"source": "test.txt", "chunk_id": "123"}),
            "cross_encoder_score": 0.9
        }
    ]
    retriever = MockRetriever(contexts)
    llm = MockLLM("Mock answer")
    builder = MockPromptBuilder()
    validator = MockEvidenceValidator(supported=True)
    
    engine = RAGEngine(retriever, llm, builder, validator)
    
    # Execute
    result = engine.ask("test query")
    
    # Verify
    assert result["supported"] is True
    assert result["answer"] == "Mock answer"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "test.txt"
    assert result["sources"][0]["chunk_id"] == "123"

def test_rag_engine_unsupported_flow():
    # Setup mocks
    retriever = MockRetriever([])
    llm = MockLLM("Should not be called")
    builder = MockPromptBuilder()
    validator = MockEvidenceValidator(supported=False)
    
    engine = RAGEngine(retriever, llm, builder, validator)
    
    # Execute
    result = engine.ask("test query")
    
    # Verify
    assert result["supported"] is False
    assert result["answer"] == "Declined."
    assert len(result["sources"]) == 0

def test_rag_engine_citation_deduplication():
    # Setup mocks with duplicate chunks
    contexts = [
        {
            "document": Document(page_content="Text", metadata={"source": "test.txt", "chunk_id": "123"})
        },
        {
            "document": Document(page_content="Text", metadata={"source": "test.txt", "chunk_id": "123"})
        }
    ]
    retriever = MockRetriever(contexts)
    llm = MockLLM("Mock answer")
    builder = MockPromptBuilder()
    validator = MockEvidenceValidator(supported=True)
    
    engine = RAGEngine(retriever, llm, builder, validator)
    
    # Execute
    result = engine.ask("test query")
    
    # Verify deduplication
    assert len(result["sources"]) == 1
