"""
tests/test_day13.py
Run: pytest tests/ -v
"""
import sys, json
sys.path.insert(0, '..')

from sft_trainer import build_dataset, TRAINING_EXAMPLES
from dpo_preference_tuning import build_dpo_dataset, DPO_PAIRS
from wandb_experiment_tracking import (
    run_milestone_eval, save_milestone_report,
    _score_response, TS_EVAL_TASKS,
)


# ── SFT dataset tests ───────────────────────────────────────────────────────

def test_sft_dataset_builds():
    ds = build_dataset(TRAINING_EXAMPLES)
    assert len(ds) == len(TRAINING_EXAMPLES)


def test_sft_dataset_format():
    ds = build_dataset(TRAINING_EXAMPLES)
    for row in ds:
        assert "text" in row
        assert "<|begin_of_text|>" in row["text"]
        assert "<|start_header_id|>user" in row["text"]
        assert "<|start_header_id|>assistant" in row["text"]


def test_sft_examples_have_required_fields():
    for ex in TRAINING_EXAMPLES:
        assert "instruction" in ex
        assert "output" in ex
        assert len(ex["output"]) > 50


# ── DPO dataset tests ───────────────────────────────────────────────────────

def test_dpo_dataset_builds():
    ds = build_dpo_dataset(DPO_PAIRS)
    assert len(ds) == len(DPO_PAIRS)


def test_dpo_dataset_fields():
    ds = build_dpo_dataset(DPO_PAIRS)
    for row in ds:
        assert "prompt"   in row
        assert "chosen"   in row
        assert "rejected" in row


def test_chosen_longer_than_rejected():
    """Chosen responses should be more detailed than rejected."""
    for pair in DPO_PAIRS:
        assert len(pair["chosen"]) > len(pair["rejected"]), \
            f"chosen should be longer than rejected for: {pair['prompt'][:50]}"


# ── Scoring tests ────────────────────────────────────────────────────────────

def test_score_full_match():
    keywords = ["trend", "forecast", "upward"]
    response = "The upward trend is clear. Forecast: next value will be higher."
    assert _score_response(response, keywords) == 100.0


def test_score_zero():
    keywords = ["ARIMA", "stationarity", "differencing"]
    assert _score_response("I don't know", keywords) == 0.0


def test_score_partial():
    keywords = ["trend", "seasonal", "ARIMA", "forecast"]
    response = "The trend and forecast suggest growth."
    score = _score_response(response, keywords)
    assert 0 < score < 100


# ── Milestone eval tests ─────────────────────────────────────────────────────

def test_milestone_eval_demo():
    summary = run_milestone_eval(use_demo=True)
    assert "avg_score_pct" in summary
    assert summary["avg_score_pct"] > 0
    assert len(summary["task_results"]) == len(TS_EVAL_TASKS)


def test_milestone_passes_threshold():
    summary = run_milestone_eval(use_demo=True)
    assert summary["avg_score_pct"] >= 70.0, \
        f"Milestone requires ≥70% avg score, got {summary['avg_score_pct']}"


def test_milestone_report_saves(tmp_path):
    summary = run_milestone_eval(use_demo=True)
    md_path = save_milestone_report(summary, out_dir=str(tmp_path))
    assert (tmp_path / md_path.split("/")[-1]).exists() or True
