"""
Day 12: ML Task Definitions
Author: Sheshikala
Date: 2026-03-12

These are the actual task functions that get queued in Redis
and executed by run_tasks.py

What I learned:
- Keep each task focused on one job
- Always return a result dict - makes it easy to store in MongoDB
- time.sleep simulates real training time
"""

import time
import random
import datetime


def train_model(payload: dict) -> dict:
    """
    Simulates training an ML model
    In real project this would call sklearn/pytorch/transformers
    """
    model_name  = payload.get("model_name", "Unknown")
    epochs      = payload.get("epochs", 3)
    learning_rate = payload.get("learning_rate", 2e-5)

    print(f"    🏋️  Training {model_name} | epochs={epochs} | lr={learning_rate}")

    # simulate training time
    time.sleep(1.5)

    # simulate metrics - slight randomness to make it realistic
    base_f1 = payload.get("expected_f1", 0.88)
    test_f1 = round(base_f1 + random.uniform(-0.01, 0.02), 4)
    loss    = round(random.uniform(0.15, 0.35), 4)

    return {
        "task_type"   : "train",
        "model_name"  : model_name,
        "status"      : "completed",
        "metrics"     : {
            "test_f1"      : test_f1,
            "test_accuracy": round(test_f1 + 0.005, 4),
            "loss"         : loss,
            "epochs_run"   : epochs
        },
        "hyperparameters": {
            "epochs"       : epochs,
            "learning_rate": learning_rate
        },
        "completed_at": datetime.datetime.now().isoformat()
    }


def evaluate_model(payload: dict) -> dict:
    """
    Simulates evaluating a trained model on test set
    """
    model_name  = payload.get("model_name", "Unknown")
    dataset     = payload.get("dataset", "test_set")

    print(f"    📊  Evaluating {model_name} on {dataset}")

    time.sleep(0.8)

    base_f1 = payload.get("expected_f1", 0.88)

    return {
        "task_type"  : "evaluate",
        "model_name" : model_name,
        "dataset"    : dataset,
        "status"     : "completed",
        "metrics"    : {
            "f1_score"      : round(base_f1 + random.uniform(-0.005, 0.01), 4),
            "precision"     : round(base_f1 + random.uniform(0.0, 0.015), 4),
            "recall"        : round(base_f1 + random.uniform(-0.01, 0.005), 4),
            "inference_ms"  : round(random.uniform(10, 150), 1)
        },
        "completed_at": datetime.datetime.now().isoformat()
    }


def generate_embeddings(payload: dict) -> dict:
    """
    Simulates generating text embeddings
    In real project this would call sentence-transformers
    """
    model_name  = payload.get("model_name", "Unknown")
    num_docs    = payload.get("num_documents", 100)

    print(f"    🔢  Generating embeddings: {model_name} | {num_docs} documents")

    time.sleep(1.0)

    return {
        "task_type"       : "embed",
        "model_name"      : model_name,
        "status"          : "completed",
        "documents_processed": num_docs,
        "embedding_dim"   : 768,
        "avg_time_per_doc": round(random.uniform(5, 25), 2),
        "completed_at"    : datetime.datetime.now().isoformat()
    }


# task registry - maps task type string to function
# run_tasks.py uses this to know which function to call
TASK_REGISTRY = {
    "train"   : train_model,
    "evaluate": evaluate_model,
    "embed"   : generate_embeddings,
}
