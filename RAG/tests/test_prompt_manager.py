import pytest
import os
from pathlib import Path
from src.prompt.prompt_manager import PromptManager

@pytest.fixture
def manager():
    return PromptManager()

def test_prompt_manager_loads_default(manager):
    # The default prompt 'rag_answer_v1.txt' should be loaded successfully
    prompt = manager.get_prompt()
    assert "{context}" in prompt
    assert "{question}" in prompt
    assert "You are a retrieval-augmented assistant." in prompt

def test_prompt_manager_caching(manager):
    # First call reads from disk
    prompt1 = manager.get_prompt("rag_answer_v1")
    
    # Second call should read from cache
    # We can verify this by checking if the version is in the _cache dict
    assert "rag_answer_v1" in manager._cache
    prompt2 = manager.get_prompt("rag_answer_v1")
    
    assert prompt1 == prompt2

def test_prompt_manager_missing_file_raises_error(manager):
    with pytest.raises(FileNotFoundError):
        manager.get_prompt("non_existent_version_v999")
