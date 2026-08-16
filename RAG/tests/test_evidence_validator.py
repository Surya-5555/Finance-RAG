import pytest
from src.generation.evidence_validator import EvidenceValidator

class MockDocument:
    def __init__(self, content):
        self.page_content = content

@pytest.fixture
def validator():
    return EvidenceValidator()

def test_evidence_passes(validator):
    contexts = [
        {"document": MockDocument("The revenue was 45 million and net income 1.2 billion.")}
    ]
    answer = "The company reported 45 million in revenue and 1.2 billion in income."
    assert validator.validate_answer(answer, contexts) is True

def test_evidence_fails_due_to_missing_number(validator):
    contexts = [
        {"document": MockDocument("The revenue was 45 million.")}
    ]
    answer = "The revenue was 45 million, representing a 10% increase."
    assert validator.validate_answer(answer, contexts) is False

def test_evidence_passes_no_numbers(validator):
    contexts = [
        {"document": MockDocument("The company is growing rapidly.")}
    ]
    answer = "The company is experiencing rapid growth."
    assert validator.validate_answer(answer, contexts) is True

def test_evidence_fails_no_chunks(validator):
    contexts = []
    answer = "Revenue was 45 million."
    assert validator.validate_answer(answer, contexts) is False

def test_decline_response(validator):
    response = validator.get_decline_response()
    assert "not available in the uploaded documents" in response
