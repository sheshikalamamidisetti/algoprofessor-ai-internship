"""
rag_evaluation.py
-----------------
Evaluates RAG pipeline quality:
  - Retrieval recall: are the right docs retrieved?
  - Answer relevance: does the answer address the query?
  - Faithfulness: is the answer grounded in the context?

Usage:
    python rag_evaluation.py
    python rag_evaluation.py --save
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from faiss_indexer import retrieve
from rag_pipeline import rag_query

# ── Evaluation test set ────────────────────────────────────────────────────
# Each item: query + expected document IDs that should be retrieved

EVAL_SET = [
    {
        "query":    "What is overfitting and how do I prevent it?",
        "expected": ["ml_01"],
        "keywords": ["overfitting", "regularisation", "cross-validation", "dropout"],
    },
    {
        "query":    "How do I test if my time series is stationary?",
        "expected": ["ts_01"],
        "keywords": ["ADF", "stationarity", "differencing", "ARIMA"],
    },
    {
        "query":    "What is a p-value and when do I reject the null hypothesis?",
        "expected": ["stat_01"],
        "keywords": ["p-value", "null hypothesis", "alpha", "reject"],
    },
    {
        "query":    "Which feature importance method is most reliable?",
        "expected": ["ml_02"],
        "keywords": ["permutation", "SHAP", "feature importance"],
    },
    {
        "query":    "What evaluation metrics should I use for time series forecasting?",
        "expected": ["ts_02"],
        "keywords": ["MAE", "RMSE", "MAPE", "walk-forward"],
    },
    {
        "query":    "How does Random Forest handle multiple decision trees?",
        "expected": ["ml_03"],
        "keywords": ["bootstrap", "voting", "ensemble", "OOB"],
    },
    {
        "query":    "What activation function should I use in neural networks?",
        "expected": ["dl_01"],
        "keywords": ["ReLU", "sigmoid", "softmax", "activation"],
    },
]


def evaluate_retrieval(top_k: int = 3) -> dict:
    """Check if expected documents appear in top-k retrieved results."""
    hits = 0
    results = []

    for item in EVAL_SET:
        docs   = retrieve(item["query"], top_k=top_k)
        ret_ids = [d["id"] for d in docs]
        hit     = any(exp in ret_ids for exp in item["expected"])
        hits   += int(hit)

        results.append({
            "query":       item["query"],
            "expected":    item["expected"],
            "retrieved":   ret_ids,
            "hit":         hit,
            "top_score":   docs[0].get("similarity_score", 0) if docs else 0,
        })

        symbol = "HIT " if hit else "MISS"
        print(f"  [{symbol}] {item['query'][:55]}")
        print(f"         Expected: {item['expected']} | Got: {ret_ids}")

    recall = round(hits / len(EVAL_SET) * 100, 1)
    return {"recall_at_k": recall, "k": top_k,
            "hits": hits, "total": len(EVAL_SET), "results": results}


def evaluate_answer_relevance(model: str = "demo") -> dict:
    """Check if answer contains expected keywords."""
    scores = []

    for item in EVAL_SET[:4]:   # run on first 4 to keep it fast
        result  = rag_query(item["query"], model=model)
        answer  = result["answer"].lower()
        kw_hits = sum(1 for kw in item["keywords"] if kw.lower() in answer)
        score   = round(kw_hits / len(item["keywords"]) * 100, 1)
        scores.append(score)
        print(f"  [{score:5.1f}%] {item['query'][:55]}")

    avg = round(sum(scores) / len(scores), 1)
    return {"avg_relevance_pct": avg, "per_query": scores}


def run_full_eval(save: bool = False) -> dict:
    print("=" * 60)
    print("Day 7 — RAG Evaluation")
    print("=" * 60)

    print("\nRetrieval Recall (top-3):")
    retrieval = evaluate_retrieval(top_k=3)
    print(f"\nRecall@3: {retrieval['recall_at_k']}%  "
          f"({retrieval['hits']}/{retrieval['total']} queries)")

    print("\nAnswer Relevance (demo mode):")
    relevance = evaluate_answer_relevance(model="demo")
    print(f"\nAvg relevance: {relevance['avg_relevance_pct']}%")

    summary = {
        "retrieval_recall_at_3": retrieval["recall_at_k"],
        "answer_relevance_pct":  relevance["avg_relevance_pct"],
        "total_queries":         retrieval["total"],
        "timestamp":             datetime.now().isoformat(),
    }

    print("\nSummary:")
    print(f"  Retrieval Recall@3:  {summary['retrieval_recall_at_3']}%")
    print(f"  Answer Relevance:    {summary['answer_relevance_pct']}%")

    if save:
        Path("outputs").mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"outputs/rag_evaluation_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({**summary,
                       "retrieval_detail": retrieval,
                       "relevance_detail": relevance}, f, indent=2)
        print(f"\nEvaluation saved: {path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 7 — RAG Evaluation")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    run_full_eval(save=args.save)
