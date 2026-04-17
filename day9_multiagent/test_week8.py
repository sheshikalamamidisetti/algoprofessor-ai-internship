"""
tests/test_week8.py
Run: pytest tests/ -v  (from day9_multiagent/)
"""
import sys, json
sys.path.insert(0, '..')

from eval_observability import (
    run_mt_bench, run_helm_summary, score_response, MT_BENCH_TASKS
)
from insightbot import run_demo_insightbot, run_milestone_eval


# ── eval_observability tests ────────────────────────────────────────────────

def test_score_full():
    rubric = ["overfitting", "feature selection", "regularisation"]
    resp = "overfitting occurs when model memorises training data, use feature selection and regularisation"
    assert score_response(resp, rubric) == 100.0


def test_score_zero():
    rubric = ["Mann-Whitney", "non-parametric"]
    assert score_response("I don't know", rubric) == 0.0


def test_mt_bench_runs():
    results = run_mt_bench(model_name="demo")
    assert len(results) == len(MT_BENCH_TASKS)
    for r in results:
        assert "task_id" in r
        assert "avg_score" in r
        assert 0 <= r["avg_score"] <= 100


def test_helm_metrics():
    mt = [{"avg_score": 80.0}, {"avg_score": 90.0}, {"avg_score": 70.0}]
    helm = run_helm_summary(mt)
    assert "accuracy" in helm
    assert "toxicity" in helm
    assert helm["toxicity"] == 0.0
    assert 0 <= helm["accuracy"] <= 100


# ── insightbot tests ────────────────────────────────────────────────────────

def test_demo_insightbot_returns_report():
    result = run_demo_insightbot("analyse churn data")
    assert "report" in result
    assert "InsightBot Report" in result["report"]
    assert result["agents_used"] == 5


def test_demo_insightbot_has_quality():
    result = run_demo_insightbot("forecast revenue")
    assert "quality" in result
    assert result["quality"] >= 7.0


def test_milestone_eval_passes():
    summary = run_milestone_eval()
    assert summary["avg_quality"] >= 7.5, \
        f"M8 requires avg quality ≥7.5, got {summary['avg_quality']}"
    assert summary["passed"] is True
    assert summary["queries_run"] == 5


def test_milestone_saves_files():
    import glob
    summary = run_milestone_eval()
    json_files = glob.glob("outputs/InsightBot_M8_*.json")
    md_files   = glob.glob("outputs/InsightBot_M8_*.md")
    assert len(json_files) >= 1, "M8 JSON report not saved"
    assert len(md_files)   >= 1, "M8 MD report not saved"


def test_milestone_agents_listed():
    summary = run_milestone_eval()
    assert "agents" in summary
    assert len(summary["agents"]) == 5
    agent_names = " ".join(summary["agents"])
    assert "DataRetriever" in agent_names
    assert "ChartAnalyst" in agent_names
    assert "StatReasoner" in agent_names
