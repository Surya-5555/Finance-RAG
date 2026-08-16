import pytest
from src.generation.evidence_validator import EvidenceValidator

@pytest.fixture
def validator():
    val = EvidenceValidator()
    # Mock config to avoid reading from disk
    val.enabled = True
    val.min_score = 0.50
    val.min_chunks = 2
    return val

def test_evidence_passes(validator):
    contexts = [
        {"cross_encoder_score": 0.8},
        {"cross_encoder_score": 0.6},
        {"cross_encoder_score": 0.2} # Ignored
    ]
    result = validator.validate(contexts)
    assert result["supported"] is True

def test_evidence_fails_due_to_low_score(validator):
    contexts = [
        {"cross_encoder_score": 0.8},
        {"cross_encoder_score": 0.3} # Below 0.5 threshold
    ]
    result = validator.validate(contexts)
    assert result["supported"] is False
    assert "Insufficient supporting evidence" in result["reason"]

def test_evidence_fails_no_chunks(validator):
    contexts = []
    result = validator.validate(contexts)
    assert result["supported"] is False
    assert "No documents retrieved" in result["reason"]

def test_evidence_validator_disabled(validator):
    validator.enabled = False
    contexts = [{"cross_encoder_score": 0.1}] # Would normally fail
    result = validator.validate(contexts)
    assert result["supported"] is True

def test_decline_response(validator):
    response = validator.get_decline_response()
    assert "I don't have enough information" in response
