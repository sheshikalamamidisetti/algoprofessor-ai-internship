"""
pytest tests — LLM Evaluator
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm_evaluator import LLMEvaluator


@pytest.fixture
def evaluator():
    return LLMEvaluator()


def test_evaluate_response_returns_dict(evaluator):
    result = evaluator.evaluate_response(
        question="What is the revenue?",
        answer="Revenue is $1.2M this quarter.",
        context="Q1 revenue reached $1.2M"
    )
    assert isinstance(result, dict)


def test_evaluate_response_has_required_keys(evaluator):
    result = evaluator.evaluate_response("What is revenue?", "Revenue is $1.2M", "")
    assert "faithfulness_score" in result
    assert "relevancy_score" in result
    assert "overall_score" in result
    assert "hallucination_risk" in result


def test_faithfulness_score_range(evaluator):
    result = evaluator.evaluate_response("Revenue?", "Revenue is $1.2M", "Revenue reached $1.2M")
    assert 0.0 <= result["faithfulness_score"] <= 1.0


def test_relevancy_score_range(evaluator):
    result = evaluator.evaluate_response("What is revenue?", "Revenue is $1.2M", "")
    assert 0.0 <= result["relevancy_score"] <= 1.0


def test_overall_score_is_average(evaluator):
    result = evaluator.evaluate_response("Revenue?", "Revenue is $1.2M", "")
    expected = round((result["faithfulness_score"] + result["relevancy_score"]) / 2, 2)
    assert result["overall_score"] == expected


def test_hallucination_risk_values(evaluator):
    result = evaluator.evaluate_response("Revenue?", "Revenue is definitely 100% guaranteed.", "")
    assert result["hallucination_risk"] in ["low", "medium", "high"]


def test_batch_evaluate(evaluator):
    qa_pairs = [
        {"question": "What is revenue?", "answer": "Revenue is $1.2M", "context": "Revenue $1.2M"},
        {"question": "What is growth?", "answer": "Growth is 15%", "context": "Growth 15%"},
    ]
    result = evaluator.batch_evaluate(qa_pairs)
    assert result["total_evaluated"] == 2
    assert "avg_overall_score" in result
    assert "avg_faithfulness" in result


def test_batch_evaluate_scores_in_range(evaluator):
    qa_pairs = [{"question": "Q1?", "answer": "A1", "context": "C1"}]
    result = evaluator.batch_evaluate(qa_pairs)
    assert 0.0 <= result["avg_overall_score"] <= 1.0
