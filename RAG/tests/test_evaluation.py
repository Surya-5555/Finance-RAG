import pytest
import os
import json
from pathlib import Path
from src.evaluation.ragas_evaluator import RagasEvaluator

# Note: We don't import or run run_evaluation.py directly to avoid running the whole
# script in pytest, but we test its components. We can import its dataset loader.
from evaluation.run_evaluation import load_dataset

@pytest.fixture
def evaluator():
    eval_obj = RagasEvaluator()
    # Mock config to avoid reading from disk
    eval_obj.enabled = True
    eval_obj.configured_metrics = ["faithfulness", "answer_relevancy"]
    return eval_obj

def test_dataset_loading(tmp_path):
    # Create a temporary JSONL file
    dataset_file = tmp_path / "test_dataset.jsonl"
    
    data = [
        {"id": "q1", "question": "Q1", "reference_answer": "A1", "expected_sources": []},
        {"id": "q2", "question": "Q2", "reference_answer": "A2", "expected_sources": []}
    ]
    
    with open(dataset_file, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
            
    loaded_data = load_dataset(str(dataset_file))
    
    assert len(loaded_data) == 2
    assert loaded_data[0]["id"] == "q1"
    assert loaded_data[1]["reference_answer"] == "A2"

def test_missing_dataset_loading():
    loaded_data = load_dataset("non_existent_file.jsonl")
    assert isinstance(loaded_data, list)
    assert len(loaded_data) == 0

def test_ragas_evaluator_metric_mapping(evaluator):
    metrics = evaluator.get_active_metrics()
    
    # It should have 2 metrics since we configured ["faithfulness", "answer_relevancy"]
    assert len(metrics) == 2
    
    # Extract names (or classes) to verify. Ragas metric objects have a `.name` attribute.
    # Note: If ragas failed to import in ragas_evaluator.py (evaluate=None), 
    # the mapping might be partially empty, but in our environment we installed ragas.
    metric_names = [m.name for m in metrics if m is not None]
    if metric_names:
        assert "faithfulness" in metric_names
        assert "answer_relevancy" in metric_names

def test_ragas_evaluator_disabled():
    eval_obj = RagasEvaluator()
    eval_obj.enabled = False
    
    result = eval_obj.evaluate([])
    assert result == {} # Should return empty dict if disabled

def test_ragas_evaluator_empty_metrics(evaluator):
    evaluator.configured_metrics = ["invalid_metric_name"]
    active = evaluator.get_active_metrics()
    assert len(active) == 0
    
    result = evaluator.evaluate([{"question": "Q", "answer": "A", "contexts": ["C"], "ground_truth": "G"}])
    assert result == {} # Should abort if no active metrics

import pytest
from evaluation.run_evaluation import validate_record

def test_dataset_validation_missing_keys():
    all_ids = set()
    record = {"id": "1", "question": "test"} # Missing expected_sources and reference_answer
    assert not validate_record(record, 0, all_ids)

def test_dataset_validation_duplicate_ids():
    all_ids = {"1"}
    record = {
        "id": "1", "question": "q", "reference_answer": "a",
        "expected_sources": [{"source": "test.txt", "page": 1}]
    }
    assert not validate_record(record, 0, all_ids)

def test_dataset_validation_empty_strings():
    all_ids = set()
    record = {
        "id": "2", "question": "   ", "reference_answer": "a",
        "expected_sources": [{"source": "test.txt", "page": 1}]
    }
    assert not validate_record(record, 0, all_ids)
    
def test_dataset_validation_bad_sources():
    all_ids = set()
    record = {
        "id": "3", "question": "q", "reference_answer": "a",
        "expected_sources": [{"wrong_key": "val"}] # Missing source
    }
    assert not validate_record(record, 0, all_ids)

def test_dataset_validation_success():
    all_ids = set()
    record = {
        "id": "4", "question": "q", "reference_answer": "a",
        "expected_sources": [{"source": "test.txt", "page": 1}]
    }
    assert validate_record(record, 0, all_ids)
