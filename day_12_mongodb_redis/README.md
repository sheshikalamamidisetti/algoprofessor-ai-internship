Day 12: MongoDB + Redis
Author: Sheshikala
Date: March 12, 2026
Theme: ML Experiment Tracker

What I Built
Worked with two NoSQL databases — MongoDB for flexible document storage and Redis for in-memory caching and task queuing. Kept the same ML Experiment Tracker theme from Day 11 so all data stays consistent across days. The main thing I understood today is that MongoDB and Redis are not replacements for PostgreSQL — they solve different problems and all three are used together in real systems.

Files
FileWhat it doesmongo_crud.pyINSERT, FIND, UPDATE, DELETE with all query operators and indexesagg_pipeline.py6 aggregation pipelines — $match, $group, $lookup, $unwind, $addFields, $facetredis_cache.pyAll 5 Redis data types + Cache-Aside pattern + Pub/Sub + TTLtasks.pyTask function definitions for train, evaluate, embed operationsrun_tasks.pyRedis LIST as task queue, processes tasks and stores results in MongoDBcache_benchmark.ipynbBenchmarks Cache-Aside vs Write-Through vs Write-Behind with chartsmongodblearning.docxMongoDB learning notes — CRUD, aggregation, indexes explainedredislearning.docxRedis learning notes — data types, caching patterns, TTL explainedresults.docxScreenshots and observations from running all filesrequirements.txtPython dependencies

How to Run
Make sure MongoDB and Redis are running, then:
bashpip install -r requirements.txt

python mongo_crud.py
python agg_pipeline.py
python redis_cache.py
python run_tasks.py

jupyter notebook cache_benchmark.ipynb
Run files in this order. Each file seeds its own data so they can run independently.

MongoDB
Why MongoDB for ML Experiments
In Day 11 with PostgreSQL, storing hyperparameters was awkward because BERT and XGBoost have completely different parameters. With MongoDB each experiment document just has whatever fields it needs — no NULL columns, no schema changes when a new model type is added.
Collections
CollectionPurposeexperimentsML runs with nested hyperparameters and metrics as JSONBresearchersResearcher profiles with department and expertise arraytask_resultsResults from Redis task queue stored for later querying
CRUD — mongo_crud.py
Practiced all four operations with realistic ML experiment data:

insert_one and insert_many — inserting with nested dicts for hyperparameters
find_one and find with projection — selecting specific fields using dot notation
update_one with $set and $push — updating nested fields and arrays without replacing whole document
delete_one and delete_many
Upsert — creates document automatically if filter finds no match
Query operators: $gt, $lt, $in, $exists, $and, $or, $regex, $elemMatch
Indexes: single field, compound, text, unique

Aggregation Pipeline — agg_pipeline.py
The pipeline processes documents through stages like an assembly line. Each stage takes output from previous stage.
PipelineStages UsedWhat it producesPipeline 1$match + $group + $sortExperiment stats per projectPipeline 2$group + $sortAverage F1 and hours per model typePipeline 3$unwind + $groupTag frequency analysisPipeline 4$lookup + $unwind + $projectExperiments joined with researcher detailsPipeline 5$addFields + $sortEfficiency score (F1 / training hours) per experimentPipeline 6$facet4 sub-pipelines in one query — status, projects, top 3, overall stats

Redis
Why Redis
Redis stores everything in RAM which makes it 1000x faster than MongoDB for reads. Measured this directly in cache_benchmark.ipynb — cache miss took 0.8 seconds (MongoDB query) but cache hit took 0.0035 seconds (Redis). That is 227x faster.
Data Types — redis_cache.py
TypeUsed ForKey ExampleStringsCache single values, JSON objects, countersbest_model, session:userHashesStore experiment fields separatelyexp:001 with name, f1, status fieldsListsFIFO task queuetask_queue:mlSetsTrack unique model names, set operationsnlp_models, cv_modelsSorted SetsF1 leaderboard with automatic rankingleaderboard:f1
Caching Patterns
Cache-Aside — most common pattern. Check Redis first, on MISS go to MongoDB and populate cache.
Pub/Sub — publisher sends training completion events, subscriber receives in real time. Needs separate Redis connection for subscriber thread.
TTL — all cache keys have expiry set with setex. Without TTL Redis memory fills up over time.
Task Queue — run_tasks.py
Simple but powerful queue using Redis LIST:

Producer pushes 5 ML tasks with RPUSH
Worker pops with LPOP — FIFO order guaranteed
Each task runs its function (train / evaluate / embed)
Results stored in MongoDB for later querying
Task timing stored in Redis Sorted Set for fast ranking

This is a simplified version of how Celery and RQ work internally.

Cache Benchmark — cache_benchmark.ipynb
Benchmarked 3 patterns on 100 ML experiments with charts:
PatternWrite LatencyRead LatencyConsistencyBest ForCache-AsideN/A (reads only)Fast on HITEventualRead-heavy, sparse accessWrite-ThroughSlower (DB + cache)Always fastStrongData must always be freshWrite-BehindFastest (cache only)FastEventualWrite-heavy workloads
Key finding: Write-Behind is fastest for writes but has data loss risk if Redis crashes before flush to MongoDB.

What I Found Difficult

$unwind creates one document per array element — confusing at first but needed before $group on array values
$lookup result comes as nested array — need $unwind again to access joined fields directly
Pub/Sub subscriber must have its own Redis connection — cannot share with publisher, wasted time on this
Understanding when to use Hash vs String — Hash is better when you need to update individual fields without rewriting whole object
TTL management — setex vs set + expire do the same thing, setex is just more concise


Key Takeaways
MongoDB, Redis, and PostgreSQL are all used together in production ML systems — not instead of each other:

PostgreSQL (Day 11) — structured relational data, complex JOINs, transactions
MongoDB (Day 12) — flexible documents, ML experiment metadata, JSONB hyperparameters
Redis (Day 12) — caching hot data, task queues, real time notifications, leaderboards
Vector DB (Day 13) — semantic search on experiment descriptions and model outputs


Connection to Other Days
Day 11 built the relational foundation. Day 12 adds flexible document storage and caching on top. Day 13 extends this with vector search on the same experiment data. Day 7 RAG pipeline will use Redis for caching and MongoDB for storing retrieved results.
