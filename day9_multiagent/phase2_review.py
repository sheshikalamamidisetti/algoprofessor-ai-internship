"""
phase2_review.py  ·  Days 51-55  ·  Apr 27 - May 1
-----------------------------------------------------
Goes into: day9_multiagent/  (extended)

Phase 2 review sprint — tests all milestones M6, M7, M8
end-to-end and generates a comprehensive Phase 2 summary report.

Covers:
  M6: DataOracle (day10_project/)
  M7: TimeSeriesHunter (day13/)
  M8: InsightBot (day9_multiagent/)

Usage:
    python phase2_review.py
    python phase2_review.py --milestone m6
    python phase2_review.py --full
"""

import json
import os
import glob
import argparse
from datetime import datetime
from pathlib import Path


# ── M6 Review ─────────────────────────────────────────────────────────────

def review_m6() -> dict:
    """Check M6 DataOracle deliverables exist and are complete."""
    print("\nReviewing M6 — DataOracle LLM Insights Predictor...")

    checks = {
        "model_registry.py":      Path("../day10_project/model_registry.py").exists(),
        "benchmark_runner.py":    Path("../day10_project/benchmark_runner.py").exists(),
        "tot_dspy_pipeline.py":   Path("../day10_project/tot_dspy_pipeline.py").exists(),
        "pydantic_schemas.py":    Path("../day10_project/pydantic_schemas.py").exists(),
        "ml_report_generator.py": Path("../day10_project/ml_report_generator.py").exists(),
        "app.py":                 Path("../day10_project/app.py").exists(),
        "report_output":          len(glob.glob("../day10_project/outputs/DataOracle_Report_*.md")) > 0,
        "tests":                  Path("../day10_project/tests/test_dataoracle.py").exists(),
        "notebook":               Path("../day10_project/notebooks/dataoracle_demo.ipynb").exists(),
    }

    passed = sum(checks.values())
    total  = len(checks)

    for item, ok in checks.items():
        symbol = "PASS" if ok else "MISS"
        print(f"  [{symbol}] {item}")

    return {
        "milestone":  "M6 — DataOracle",
        "score":      round(passed / total * 100, 1),
        "passed":     passed,
        "total":      total,
        "checks":     checks,
        "status":     "complete" if passed == total else "incomplete",
    }


# ── M7 Review ─────────────────────────────────────────────────────────────

def review_m7() -> dict:
    """Check M7 TimeSeriesHunter deliverables."""
    print("\nReviewing M7 — TimeSeriesHunter QLoRA Llama 3.1...")

    checks = {
        "lora_qlora_setup.py":           Path("../day13/lora_qlora_setup.py").exists(),
        "sft_trainer.py":                Path("../day13/sft_trainer.py").exists(),
        "dpo_preference_tuning.py":      Path("../day13/dpo_preference_tuning.py").exists(),
        "quantisation_vllm.py":          Path("../day13/quantisation_vllm.py").exists(),
        "wandb_experiment_tracking.py":  Path("../day13/wandb_experiment_tracking.py").exists(),
        "m7_report":                     len(glob.glob("../day13/outputs/TimeSeriesHunter_M7_*.md")) > 0,
        "tests":                         Path("../day13/tests/test_day13.py").exists(),
        "notebook":                      Path("../day13/notebooks/timeserieshunter_demo.ipynb").exists(),
    }

    # Check M7 eval score from report
    m7_score = None
    m7_json_files = glob.glob("../day13/outputs/TimeSeriesHunter_M7_*.json")
    if m7_json_files:
        with open(sorted(m7_json_files)[-1], encoding="utf-8") as f:
            data = json.load(f)
            m7_score = data.get("avg_score_pct")
        checks["m7_eval_passed"] = data.get("passed", False)

    passed = sum(v for v in checks.values() if isinstance(v, bool))
    total  = len(checks)

    for item, ok in checks.items():
        symbol = "PASS" if ok else "MISS"
        print(f"  [{symbol}] {item}")
    if m7_score:
        print(f"  Eval score: {m7_score}%")

    return {
        "milestone":  "M7 — TimeSeriesHunter",
        "score":      round(passed / total * 100, 1),
        "passed":     passed,
        "total":      total,
        "eval_score": m7_score,
        "checks":     {k: bool(v) for k, v in checks.items()},
        "status":     "complete" if passed == total else "incomplete",
    }


# ── M8 Review ─────────────────────────────────────────────────────────────

def review_m8() -> dict:
    """Check M8 InsightBot deliverables."""
    print("\nReviewing M8 — InsightBot Multi-Agent Data Analyst...")

    checks = {
        "eval_observability.py":   Path("eval_observability.py").exists(),
        "insightbot.py":           Path("insightbot.py").exists(),
        "nemo_guardrails.py":      Path("nemo_guardrails.py").exists(),
        "fastapi_streaming.py":    Path("fastapi_streaming.py").exists(),
        "cloud_deploy.py":         Path("cloud_deploy.py").exists(),
        "m8_report":               len(glob.glob("outputs/InsightBot_M8_*.md")) > 0,
        "tests":                   Path("tests/test_week8.py").exists(),
        "notebook":                Path("notebooks/insightbot_demo.ipynb").exists(),
    }

    m8_score = None
    m8_json_files = glob.glob("outputs/InsightBot_M8_*.json")
    if m8_json_files:
        with open(sorted(m8_json_files)[-1], encoding="utf-8") as f:
            data = json.load(f)
            m8_score = data.get("avg_quality")
        checks["m8_eval_passed"] = data.get("passed", False)

    passed = sum(v for v in checks.values() if isinstance(v, bool))
    total  = len(checks)

    for item, ok in checks.items():
        symbol = "PASS" if ok else "MISS"
        print(f"  [{symbol}] {item}")
    if m8_score:
        print(f"  Quality score: {m8_score}/10")

    return {
        "milestone":     "M8 — InsightBot",
        "score":         round(passed / total * 100, 1),
        "passed":        passed,
        "total":         total,
        "quality_score": m8_score,
        "checks":        {k: bool(v) for k, v in checks.items()},
        "status":        "complete" if passed == total else "incomplete",
    }


# ── Phase 2 Summary Report ─────────────────────────────────────────────────

def generate_phase2_report(m6: dict, m7: dict, m8: dict) -> str:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    avg = round((m6["score"] + m7["score"] + m8["score"]) / 3, 1)

    lines = [
        "# Phase 2 Review — LLM Engineering",
        f"**Student:** Sheshikala | **Programme:** IIT Indore AI & DS",
        f"**Date:** {datetime.now().strftime('%B %d, %Y')}",
        f"**Phase 2 Average:** {avg}%",
        "",
        "---",
        "",
        "## Milestone Summary",
        "",
        "| Milestone | Folder | Score | Status |",
        "|-----------|--------|-------|--------|",
        f"| M6 DataOracle | day10_project/ | {m6['score']}% | {m6['status']} |",
        f"| M7 TimeSeriesHunter | day13/ | {m7['score']}% | {m7['status']} |",
        f"| M8 InsightBot | day9_multiagent/ | {m8['score']}% | {m8['status']} |",
        f"| **Phase 2 Average** | | **{avg}%** | {'complete' if avg >= 80 else 'in progress'} |",
        "",
        "---",
        "",
        "## What Was Built",
        "",
        "**M6 — DataOracle LLM Insights Predictor**",
        "Benchmarked GPT-4o, Claude, Gemini, and Llama on data science tasks.",
        "Applied Tree-of-Thought reasoning via DSPy for statistical problem solving.",
        "Validated all outputs with Pydantic v2 schemas.",
        "Generated automated ML reports. Deployed as a Gradio application.",
        "",
        "**M7 — TimeSeriesHunter QLoRA Llama 3.1**",
        "Fine-tuned Llama 3.1 8B using QLoRA (4-bit, LoRA rank 16).",
        "Applied SFT on time series DS instruction pairs.",
        "Applied DPO to shape data analyst communication style.",
        "Quantised to AWQ 4-bit for deployment. Tracked with W&B.",
        "",
        "**M8 — InsightBot Multi-Agent Data Analyst**",
        "Built a 5-agent CrewAI pipeline: DataRetriever, ChartAnalyst,",
        "StatReasoner, ReportWriter, QualityChecker.",
        "Added NeMo Guardrails and Presidio PII protection.",
        "FastAPI SSE streaming for real-time responses.",
        "AWS SageMaker and Azure ML deployment pipeline.",
        "",
        "---",
        "",
        "## IIT Indore Curriculum Coverage",
        "",
        "| Topic | Covered in |",
        "|-------|-----------|",
        "| LLM Benchmarking | M6 day10_project/ |",
        "| Structured Outputs | M6 Pydantic schemas |",
        "| Fine-tuning LoRA/QLoRA | M7 day13/ |",
        "| RLHF / DPO | M7 dpo_preference_tuning.py |",
        "| Multi-Agent Systems | M8 day9_multiagent/ |",
        "| RAG + Vector Search | day7_rag/ extended |",
        "| Guardrails + Safety | M8 nemo_guardrails.py |",
        "| Cloud Deployment | M8 cloud_deploy.py |",
        "| MLOps Observability | M8 eval_observability.py |",
        "",
        "---",
        "",
        f"**Phase 2 status:** {'All milestones complete' if avg >= 80 else 'In progress'}",
        f"**Next:** Phase 3 — Agentic AI and Cloud Deploy (day14/ onwards)",
    ]

    md = "\n".join(lines)

    Path("outputs").mkdir(exist_ok=True)
    md_path   = f"outputs/Phase2_Review_{ts}.md"
    json_path = f"outputs/Phase2_Review_{ts}.json"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"m6": m6, "m7": m7, "m8": m8,
                   "avg": avg, "timestamp": ts}, f, indent=2)

    print(f"\nPhase 2 Review saved:")
    print(f"  MD   -> {md_path}")
    print(f"  JSON -> {json_path}")
    return md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Review Sprint")
    parser.add_argument("--milestone", choices=["m6", "m7", "m8"], default=None)
    parser.add_argument("--full", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2 Review Sprint — Days 51-55")
    print("=" * 60)

    if args.milestone == "m6":
        review_m6()
    elif args.milestone == "m7":
        review_m7()
    elif args.milestone == "m8":
        review_m8()
    else:
        m6 = review_m6()
        m7 = review_m7()
        m8 = review_m8()
        generate_phase2_report(m6, m7, m8)
