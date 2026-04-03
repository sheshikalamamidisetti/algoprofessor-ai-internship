"""
benchmark_runner.py  ·  Day 32
--------------------------------
Runs DS benchmark tasks across all LLMs.
Saves results to outputs/benchmark_<timestamp>.json + .csv

Usage:
    python benchmark_runner.py
    python benchmark_runner.py --models gpt4o claude
    python benchmark_runner.py --tasks stat_inference ml_code
"""

import argparse, csv, json, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from model_registry import get_model

# ── Task bank ──────────────────────────────────────────────────────────────

TASKS = {
    "stat_inference": [
        {
            "id": "si_01", "category": "stat_inference",
            "prompt": (
                "Drug trial: treatment n=50 mean=7.2d sd=1.8, control n=50 mean=8.5d sd=2.1. "
                "Choose the correct test, state H0, interpret p=0.03 at alpha=0.05."
            ),
            "rubric": ["test choice", "null hypothesis", "p-value", "conclusion"],
        },
        {
            "id": "si_02", "category": "stat_inference",
            "prompt": (
                "200 customers, 130 prefer Product A. "
                "Calculate 95% CI for true proportion. Show formula + interpret."
            ),
            "rubric": ["formula", "calculation", "CI bounds", "plain English"],
        },
        {
            "id": "si_03", "category": "stat_inference",
            "prompt": "ANOVA on 3 regions gives F=4.87, p=0.01. Interpret and name the post-hoc test.",
            "rubric": ["ANOVA interpretation", "post-hoc named", "reasoning"],
        },
    ],
    "ml_code": [
        {
            "id": "ml_01", "category": "ml_code",
            "prompt": (
                "Write a complete sklearn Pipeline for Iris classification: "
                "imputer + scaler + RandomForest + cross-validation. Full runnable code."
            ),
            "rubric": ["Pipeline", "imputer", "scaler", "RandomForest", "cross_val_score"],
        },
        {
            "id": "ml_02", "category": "ml_code",
            "prompt": "Train acc=0.99, test acc=0.67. Name the problem precisely. Give 3 fixes with code.",
            "rubric": ["overfitting", "fix 1", "fix 2", "fix 3"],
        },
    ],
    "eda": [
        {
            "id": "eda_01", "category": "eda",
            "prompt": (
                "DataFrame: 10000 rows, cols age/income/gender/churn. "
                "Write full pandas EDA: shape, dtypes, nulls, distributions, correlation, churn by gender."
            ),
            "rubric": ["shape", "null check", "distributions", "correlation", "groupby churn"],
        },
    ],
}


def _score(response: str, rubric: list[str]) -> float:
    r = response.lower()
    hits = sum(1 for item in rubric
               if sum(1 for w in item.lower().split() if w in r) >= max(1, len(item.split())//2))
    return round(hits / len(rubric) * 100, 1)


def run(model_names: list[str], categories: list[str]) -> list[dict]:
    tasks = [t for c in categories for t in TASKS.get(c, [])]
    results = []
    for mname in model_names:
        try:
            model = get_model(mname)
        except Exception as e:
            print(f"Skip {mname}: {e}"); continue
        for task in tasks:
            try:
                resp = model.chat(task["prompt"])
                score = _score(resp.response, task["rubric"])
                results.append({
                    "model": mname, "task_id": task["id"],
                    "category": task["category"], "score_pct": score,
                    "latency_ms": resp.latency_ms, "tokens": resp.tokens,
                    "response": resp.response,
                })
                print(f"  {mname}/{task['id']} → {score}% ({resp.latency_ms}ms)")
            except Exception as e:
                print(f"  ERROR {mname}/{task['id']}: {e}")
                results.append({"model": mname, "task_id": task["id"],
                                 "category": task["category"], "score_pct": 0.0,
                                 "error": str(e)})
    return results


def save(results: list[dict], out_dir: str = "outputs") -> str:
    Path(out_dir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = f"{out_dir}/benchmark_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    csv_path = f"{out_dir}/benchmark_{ts}.csv"
    fields = ["model", "task_id", "category", "score_pct", "latency_ms", "tokens"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)

    print(f"\nSaved → {json_path}")
    print(f"Saved → {csv_path}")
    return json_path


def summarise(results: list[dict]):
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r["score_pct"])
    print("\n── Benchmark Summary ───────────────────────────")
    print(f"{'Model':<12} {'Avg Score':>10} {'Tasks':>7}")
    print("-" * 32)
    for m, scores in sorted(by_model.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"{m:<12} {sum(scores)/len(scores):>9.1f}% {len(scores):>6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt4o", "claude", "gemini", "llama"])
    parser.add_argument("--tasks", nargs="+", default=["stat_inference", "ml_code", "eda"])
    args = parser.parse_args()

    print(f"Models: {args.models} | Tasks: {args.tasks}\n")
    results = run(args.models, args.tasks)
    if results:
        save(results)
        summarise(results)
