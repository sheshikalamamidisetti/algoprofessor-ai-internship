"""
Day 12: Redis Task Queue + MongoDB Result Storage
Author: Sheshikala
Date: 2026-03-12

What this does:
- Pushes ML tasks into a Redis queue (LIST)
- Worker picks tasks one by one and runs them
- Results get stored in MongoDB

What I learned:
- Redis LIST as queue = very simple but powerful pattern
- RPUSH to add, LPOP to consume = FIFO order
- Storing results in MongoDB makes them searchable later
- This is a basic version of Celery / RQ task queues

How to run:
  1. docker run -d -p 27017:27017 --name mongodb mongo
  2. docker run -d -p 6379:6379 --name redis redis
  3. pip install pymongo redis
  4. python run_tasks.py
"""

import redis
import json
import time
import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from tasks import TASK_REGISTRY

# ─────────────────────────────────────────────
# CONNECTIONS
# ─────────────────────────────────────────────
def get_connections():
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        print("✅ Redis connected!")
    except redis.ConnectionError:
        print("❌ Redis not running! Run: docker run -d -p 6379:6379 --name redis redis")
        exit(1)

    try:
        client  = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db = client["ml_tracker"]
        print("✅ MongoDB connected!")
    except ConnectionFailure:
        print("❌ MongoDB not running! Run: docker run -d -p 27017:27017 --name mongodb mongo")
        exit(1)

    return r, db

QUEUE_KEY  = "task_queue:ml"
RESULT_KEY = "task_results:ml"


# ─────────────────────────────────────────────
# PRODUCER - pushes tasks to Redis queue
# ─────────────────────────────────────────────
def enqueue_tasks(r):
    print("\n── Enqueuing Tasks ─────────────────────")

    tasks = [
        {
            "task_id"     : "task_001",
            "task_type"   : "train",
            "model_name"  : "BERT_v2",
            "epochs"      : 3,
            "learning_rate": 2e-5,
            "expected_f1" : 0.90,
            "queued_at"   : datetime.datetime.now().isoformat()
        },
        {
            "task_id"     : "task_002",
            "task_type"   : "train",
            "model_name"  : "RoBERTa_v1",
            "epochs"      : 5,
            "learning_rate": 1e-5,
            "expected_f1" : 0.93,
            "queued_at"   : datetime.datetime.now().isoformat()
        },
        {
            "task_id"     : "task_003",
            "task_type"   : "evaluate",
            "model_name"  : "ResNet50_medical",
            "dataset"     : "medical_test_set",
            "expected_f1" : 0.85,
            "queued_at"   : datetime.datetime.now().isoformat()
        },
        {
            "task_id"     : "task_004",
            "task_type"   : "embed",
            "model_name"  : "sentence-transformers",
            "num_documents": 500,
            "queued_at"   : datetime.datetime.now().isoformat()
        },
        {
            "task_id"     : "task_005",
            "task_type"   : "train",
            "model_name"  : "XGBoost_v3",
            "epochs"      : 1,
            "learning_rate": 0.05,
            "expected_f1" : 0.92,
            "queued_at"   : datetime.datetime.now().isoformat()
        },
    ]

    # push all tasks to right side of Redis list
    for task in tasks:
        r.rpush(QUEUE_KEY, json.dumps(task))
        print(f"  Queued: {task['task_id']} | type: {task['task_type']} | model: {task['model_name']}")

    print(f"\n  Total tasks in queue: {r.llen(QUEUE_KEY)}")


# ─────────────────────────────────────────────
# CONSUMER - processes tasks from queue
# ─────────────────────────────────────────────
def process_tasks(r, db):
    print("\n── Processing Tasks ────────────────────")
    col         = db["task_results"]
    completed   = 0
    failed      = 0

    while r.llen(QUEUE_KEY) > 0:
        # LPOP removes and returns leftmost task (FIFO)
        raw_task = r.lpop(QUEUE_KEY)
        if not raw_task:
            break

        task = json.loads(raw_task)
        task_id   = task["task_id"]
        task_type = task["task_type"]

        print(f"\n  ▶ Running {task_id} ({task_type})...")

        try:
            # get the right function from registry
            task_fn = TASK_REGISTRY.get(task_type)
            if not task_fn:
                raise ValueError(f"Unknown task type: {task_type}")

            start_time = time.time()
            result     = task_fn(task)
            elapsed    = round(time.time() - start_time, 2)

            # add metadata to result
            result["task_id"]    = task_id
            result["elapsed_sec"] = elapsed

            # store result in MongoDB
            col.insert_one(result)

            # also store summary in Redis sorted set (score = elapsed time)
            r.zadd(RESULT_KEY, {task_id: elapsed})

            print(f"    ✅ Done in {elapsed}s | f1: {result.get('metrics', {}).get('test_f1', 'N/A')}")
            completed += 1

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            failed += 1

    return completed, failed


# ─────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────
def show_results(r, db):
    print("\n── Results Summary ─────────────────────")
    col = db["task_results"]

    total = col.count_documents({})
    print(f"  Results stored in MongoDB: {total}")

    print("\n  Completed tasks:")
    for doc in col.find({}, {"task_id": 1, "task_type": 1, "model_name": 1, "metrics": 1, "elapsed_sec": 1, "_id": 0}):
        f1 = doc.get("metrics", {}).get("test_f1") or doc.get("metrics", {}).get("f1_score", "N/A")
        print(f"    {doc['task_id']} | {doc['task_type']:8s} | {doc['model_name']:25s} | f1: {f1} | time: {doc.get('elapsed_sec')}s")

    # task timing from Redis sorted set
    print("\n  Task timing from Redis (sorted by speed):")
    for task_id, elapsed in r.zrange(RESULT_KEY, 0, -1, withscores=True):
        print(f"    {task_id} → {elapsed}s")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    r, db = get_connections()

    # clean previous results
    r.delete(QUEUE_KEY, RESULT_KEY)
    db["task_results"].drop()

    enqueue_tasks(r)
    completed, failed = process_tasks(r, db)

    print(f"\n── Final Stats ─────────────────────────")
    print(f"  Completed : {completed}")
    print(f"  Failed    : {failed}")
    show_results(r, db)

    print("\n✅ run_tasks.py completed!")
