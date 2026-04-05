"""
tests/test_dataoracle.py
Run: pytest tests/ -v
"""
import sys
sys.path.insert(0, '..')

import pytest
from pydantic_schemas import (HypothesisTestResult, ModelMetrics,
                               LLMBenchmarkEntry, DataOracleReport)
from benchmark_runner import _score
from ml_report_generator import _default_insights


def test_hypothesis_valid():
    t = HypothesisTestResult(
        test_name="t-test", null_hypothesis="No diff",
        alternative_hypothesis="Treatment better",
        p_value=0.03, reject_null=True,
        conclusion="Reject H0", assumptions_met=["normality"])
    assert t.reject_null is True
    assert 0 <= t.p_value <= 1


def test_hypothesis_bad_p():
    with pytest.raises(Exception):
        HypothesisTestResult(
            test_name="t-test", null_hypothesis="H0",
            alternative_hypothesis="H1", p_value=1.5,
            reject_null=True, conclusion="bad", assumptions_met=[])


def test_model_metrics():
    m = ModelMetrics(accuracy=0.91, f1_score=0.88)
    assert m.accuracy == 0.91


def test_benchmark_entry():
    e = LLMBenchmarkEntry(llm="gpt4o", task_id="si_01",
                          category="stat_inference", score_pct=80.0)
    assert e.score_pct == 80.0


def test_report_markdown():
    r = DataOracleReport(
        llm_benchmarks=[
            LLMBenchmarkEntry(llm="claude", task_id="si_01",
                              category="stat_inference", score_pct=85.0)],
        best_llm="claude", best_llm_reason="Highest score",
        key_findings=["Claude best"], recommendations=["Use Claude"],
        tools_used=["Claude", "DSPy"])
    md = r.to_markdown()
    assert "DataOracle" in md
    assert "claude" in md


def test_score_full():
    rubric = ["test choice", "null hypothesis", "conclusion"]
    resp = "we use test choice, null hypothesis states no diff, conclusion is reject"
    assert _score(resp, rubric) == 100.0


def test_score_zero():
    rubric = ["ANOVA", "Tukey HSD"]
    assert _score("I don't know", rubric) == 0.0


def test_default_insights_picks_best():
    benchmarks = [
        LLMBenchmarkEntry(llm="claude", task_id="t1",
                          category="stat_inference", score_pct=90.0),
        LLMBenchmarkEntry(llm="gpt4o", task_id="t2",
                          category="ml_code", score_pct=70.0),
    ]
    ins = _default_insights(benchmarks)
    assert ins["best_llm"] == "claude"
    assert len(ins["key_findings"]) >= 3


def test_report_no_benchmarks():
    ins = _default_insights([])
    assert "best_llm" in ins
