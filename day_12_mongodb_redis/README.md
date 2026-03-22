
# Day 12: MongoDB and Redis — ML Experiment Tracker

**Author:** Sheshikala
**Date:** March 12, 2026

---

## Overview

This project extends the ML Experiment Tracker by integrating **MongoDB** and **Redis**. MongoDB is used for flexible document storage, while Redis is used for caching and task queuing.

> **Key Insight:** MongoDB, Redis, and PostgreSQL are not alternatives to each other. Each serves a different purpose and they are commonly used together in production systems.

---

## What Was Implemented

- CRUD operations and indexing in MongoDB
- Aggregation pipelines for analytical queries
- Redis caching using multiple strategies
- Task queue implementation using Redis lists
- Benchmarking of caching approaches

---

## Project Structure

| File | Description |
|---|---|
| `mongo_crud.py` | Insert, find, update, and delete operations with query operators and indexes |
| `agg_pipeline.py` | Six aggregation pipelines |
| `redis_cache.py` | Redis data types, caching patterns, Pub/Sub, and TTL |
| `tasks.py` | Training, evaluation, and embedding task definitions |
| `run_tasks.py` | Redis-based task queue implementation |
| `cache_benchmark.ipynb` | Benchmarks for different caching strategies |
| `mongodblearning.docx` | Notes on MongoDB concepts |
| `redislearning.docx` | Notes on Redis concepts |
| `results.docx` | Screenshots and observations |
| `requirements.txt` | Project dependencies |

---

## How to Run

Install dependencies and run the scripts in sequence:

```bash
pip install -r requirements.txt

python mongo_crud.py
python agg_pipeline.py
python redis_cache.py
python run_tasks.py

jupyter notebook cache_benchmark.ipynb
```

> Each script initializes its own data and can run independently.

---

## MongoDB

### Rationale

MongoDB was chosen to handle ML experiment data due to its **flexible schema**. Different models such as BERT and XGBoost have varying hyperparameters, which makes rigid relational schemas less suitable.

### Collections

| Collection | Description |
|---|---|
| `experiments` | Stores experiment data including hyperparameters and metrics |
| `researchers` | Stores researcher information |
| `task_results` | Stores results from Redis task execution |

### CRUD Operations (`mongo_crud.py`)

- Insert: `insert_one`, `insert_many`
- Read: `find`, `find_one` with projection
- Update: `$set`, `$push`
- Delete: `delete_one`, `delete_many`
- Upsert functionality

**Query operators used:** `$gt`, `$lt`, `$in`, `$exists`, `$and`, `$or`, `$regex`, `$elemMatch`

**Indexes implemented:** single-field, compound, text, and unique indexes

### Aggregation Pipelines (`agg_pipeline.py`)

The aggregation pipeline processes documents through multiple stages. Pipelines implemented:

1. Experiment statistics grouped by project
2. Average F1 score and training time by model type
3. Tag frequency analysis
4. Join between experiments and researchers using `$lookup`
5. Efficiency calculation based on F1 score and training time
6. Multi-analysis using `$facet`

---

## Redis

### Rationale

Redis was used for caching and real-time task handling due to its **in-memory architecture**. It significantly reduces read latency compared to database queries.

### Data Types (`redis_cache.py`)

| Type | Usage |
|---|---|
| Strings | Caching values |
| Hashes | Storing structured experiment data |
| Lists | Task queues |
| Sets | Tracking unique model names |
| Sorted Sets | Ranking experiments |

### Caching Strategies

| Strategy | Description |
|---|---|
| **Cache-Aside** | Data is loaded into cache only when needed |
| **Write-Through** | Data is written to both cache and database simultaneously |
| **Write-Behind** | Data is written to cache first and persisted later |

TTL was used to ensure cache expiration and prevent memory overflow.

### Task Queue (`run_tasks.py`)

A simple queue implemented using Redis lists:

- Tasks are pushed using `RPUSH`
- Workers process tasks using `LPOP`
- Tasks include training, evaluation, and embedding
- Results are stored in MongoDB
- Execution time is tracked using Redis sorted sets

---

## Cache Benchmark

Three caching strategies were benchmarked using **100 ML experiments**:

- Cache-Aside
- Write-Through
- Write-Behind

> **Note:** Write-Behind showed the fastest write performance but carries a risk of data loss if Redis fails before persistence.

---

## Challenges

- Understanding `$unwind` behavior with arrays
- Handling nested results from `$lookup`
- Managing separate Redis connections for Pub/Sub
- Choosing between Hash and String data types
- Implementing TTL correctly

---

## Key Takeaways

| Technology | Best For |
|---|---|
| **PostgreSQL** | Structured data and transactions |
| **MongoDB** | Flexible and semi-structured data |
| **Redis** | Caching, queues, and real-time operations |
