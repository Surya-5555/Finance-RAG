import pytest
from langchain_core.documents import Document
from src.prompt.builder import PromptBuilder
from src.prompt.prompt_manager import PromptManager

@pytest.fixture
def builder():
    return PromptBuilder(prompt_manager=PromptManager())

def test_prompt_builder_format_context(builder):
    contexts = [
        {
            "document": Document(page_content="Apple is a fruit.", metadata={"source": "doc1.txt", "page": 1})
        },
        {
            "document": Document(page_content="Bananas are yellow.", metadata={"filename": "doc2.pdf", "page": 2})
        }
    ]
    
    formatted = builder.format_context(contexts)
    
    assert "[Source 1]" in formatted
    assert "File: doc1.txt" in formatted
    assert "Page: 1" in formatted
    assert "Apple is a fruit." in formatted
    
    assert "[Source 2]" in formatted
    assert "File: doc2.pdf" in formatted
    assert "Page: 2" in formatted
    assert "Bananas are yellow." in formatted

def test_prompt_builder_build(builder):
    contexts = [
        {
            "document": Document(page_content="Apple is a fruit.", metadata={"source": "doc1.txt"})
        }
    ]
    query = "What is an apple?"
    
    final_prompt = builder.build(query, contexts)
    
    # Check that system instructions exist
    assert "You are a retrieval-augmented assistant." in final_prompt
    
    # Check context injection
    assert "[Source 1]" in final_prompt
    assert "Apple is a fruit." in final_prompt
    
    # Check query injection
    assert "What is an apple?" in final_prompt
