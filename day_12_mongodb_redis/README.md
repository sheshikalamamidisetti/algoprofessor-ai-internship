# Day 12: MongoDB + Redis
**Author:** Sheshikala  
**Date:** March 12, 2026

---

## What I Built

For Day 12 I worked with two NoSQL databases — MongoDB for document storage and Redis for caching and queuing. I kept the same ML Experiment Tracker theme from Day 11 so all the data stays consistent across days.

---



## How to Run

### Start Docker services first:
```bash
docker run -d -p 27017:27017 --name mongodb mongo
docker run -d -p 6379:6379 --name redis redis
```

### Install packages:
```bash
pip install -r requirements.txt
```

### Run in this order:
```bash
python mongo_crud.py       # Step 1: insert and query experiments
python agg_pipeline.py     # Step 2: aggregation pipelines
python redis_cache.py      # Step 3: Redis data types
python run_tasks.py        # Step 4: task queue
```

### For notebook:
```bash
jupyter notebook cache_benchmark.ipynb
```

---

## MongoDB — What I Practiced

### Collections Used
- `experiments` — ML experiment documents with nested hyperparameters and metrics
- `researchers` — researcher profiles with department info
- `task_results` — results from Redis task queue

### CRUD (mongo_crud.py)
- `insert_one` and `insert_many` — inserting with nested dicts for hyperparameters
- `find_one` and `find` with projection — selecting specific fields
- `update_one` with `$set` and `$push` — updating nested fields and arrays
- `delete_one` and `delete_many`
- Upsert — creates document if it doesn't exist
- Query operators: `$gt`, `$lt`, `$in`, `$exists`, `$and`, `$regex`, `$elemMatch`

### Aggregation (agg_pipeline.py)
| Pipeline | What it does |
|---|---|
| Pipeline 1 | `$match` + `$group` + `$sort` — stats by project |
| Pipeline 2 | Model type comparison — avg F1 per model |
| Pipeline 3 | `$unwind` — flatten tags array to count frequency |
| Pipeline 4 | `$lookup` — JOIN experiments with researchers |
| Pipeline 5 | `$addFields` — compute performance tier and efficiency score |
| Pipeline 6 | `$facet` — run 4 sub-pipelines in one query |

---

## Redis — What I Practiced

### Data Types (redis_cache.py)
| Type | Used For |
|---|---|
| Strings | Cache single values, JSON objects, counters |
| Hashes | Store experiment objects field by field |
| Lists | Training task queue (FIFO) |
| Sets | Track unique model names, set operations |
| Sorted Sets | F1 leaderboard with automatic ranking |

### Patterns
- **Cache-Aside** — check Redis first, on miss go to MongoDB
- **Pub/Sub** — real time notifications between processes
- **TTL** — all cache keys have expiry to avoid memory issues

### Task Queue (run_tasks.py)
- Tasks pushed to Redis LIST with `RPUSH`
- Worker pops with `LPOP` and runs the task
- Results stored in MongoDB for querying later

---

## Cache Benchmark (notebook)

Compared 3 caching patterns on 100 ML experiments:

| Pattern | Write Speed | Consistency | Best For |
|---|---|---|---|
| Cache-Aside | N/A (read only) | Eventual | Read-heavy |
| Write-Through | Slower (DB + cache) | Strong | Always fresh data |
| Write-Behind | Fastest (cache only) | Eventual | Write-heavy |

---

## What I Found Difficult

- `$unwind` was confusing at first — it creates one document per array element
- `$lookup` + `$unwind` together felt like SQL JOIN but different syntax
- Understanding when to use Hash vs String for caching objects
- Pub/Sub needs a separate Redis connection in a thread — took a while to understand why

---

## Connection to Other Days

- Day 11 → same ML Experiment Tracker theme, now adding document storage
- Day 13 → vector_db experiments stored here will connect to Chroma/FAISS
- Day 14 → RAG pipeline will use Redis for caching and MongoDB for results

