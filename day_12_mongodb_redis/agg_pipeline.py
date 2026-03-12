"""
Day 12: MongoDB Aggregation Pipeline
Author: Sheshikala
Date: 2026-03-12

What I learned:
- Aggregation pipeline = process data in stages like an assembly line
- Each stage transforms data and passes result to next stage
- Always put $match first - filters early = much faster
- $group is like GROUP BY in SQL
- $lookup is like JOIN - connects two collections
- $unwind flattens arrays into separate documents
- $facet runs multiple pipelines in one query - very powerful

How to run:
  1. Start MongoDB: docker run -d -p 27017:27017 --name mongodb mongo
  2. pip install pymongo
  3. python agg_pipeline.py  (seeds its own data)
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import datetime

# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
def get_db():
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        print("✅ Connected to MongoDB!")
        return client["ml_tracker"]
    except ConnectionFailure:
        print("❌ MongoDB not running! Run: docker run -d -p 27017:27017 --name mongodb mongo")
        exit(1)


# ─────────────────────────────────────────────
# SEED DATA
# ─────────────────────────────────────────────
def seed_data(db):
    print("\n── Seeding Data ────────────────────────")
    db["experiments"].drop()
    db["researchers"].drop()

    db["experiments"].insert_many([
        {
            "name": "BERT Sentiment", "model_type": "BERT",
            "project": "Sentiment Analysis", "researcher": "Sheshikala",
            "status": "completed", "training_hours": 2.5,
            "metrics": {"test_f1": 0.9020, "test_accuracy": 0.9050, "loss": 0.261},
            "tags": ["NLP", "Transformer"],
            "created_at": datetime.datetime(2026, 3, 1)
        },
        {
            "name": "RoBERTa Sentiment", "model_type": "RoBERTa",
            "project": "Sentiment Analysis", "researcher": "Sheshikala",
            "status": "completed", "training_hours": 4.2,
            "metrics": {"test_f1": 0.9360, "test_accuracy": 0.9380, "loss": 0.181},
            "tags": ["NLP", "Transformer"],
            "created_at": datetime.datetime(2026, 3, 2)
        },
        {
            "name": "DistilBERT Fast", "model_type": "DistilBERT",
            "project": "Sentiment Analysis", "researcher": "Sheshikala",
            "status": "completed", "training_hours": 1.5,
            "metrics": {"test_f1": 0.8985, "test_accuracy": 0.9010, "loss": 0.272},
            "tags": ["NLP", "Transformer", "Lightweight"],
            "created_at": datetime.datetime(2026, 3, 3)
        },
        {
            "name": "ResNet50 Medical", "model_type": "ResNet50",
            "project": "Medical Imaging", "researcher": "Sheshikala",
            "status": "completed", "training_hours": 10.0,
            "metrics": {"test_f1": 0.8530, "test_accuracy": 0.8580, "loss": 0.394},
            "tags": ["CV", "Transfer Learning"],
            "created_at": datetime.datetime(2026, 3, 4)
        },
        {
            "name": "DenseNet Medical", "model_type": "DenseNet121",
            "project": "Medical Imaging", "researcher": "ravi",
            "status": "completed", "training_hours": 12.0,
            "metrics": {"test_f1": 0.8780, "test_accuracy": 0.8820, "loss": 0.338},
            "tags": ["CV", "Transfer Learning"],
            "created_at": datetime.datetime(2026, 3, 5)
        },
        {
            "name": "XGBoost Fraud", "model_type": "XGBoost",
            "project": "Fraud Detection", "researcher": "Sheshikala",
            "status": "completed", "training_hours": 0.75,
            "metrics": {"test_f1": 0.9279, "test_accuracy": 0.9980, "loss": 0.072},
            "tags": ["Tabular", "Ensemble"],
            "created_at": datetime.datetime(2026, 3, 6)
        },
        {
            "name": "LightGBM Fraud", "model_type": "LightGBM",
            "project": "Fraud Detection", "researcher": "priya",
            "status": "running", "training_hours": 0.5,
            "metrics": {"test_f1": 0.9100, "test_accuracy": 0.9970},
            "tags": ["Tabular", "Ensemble"],
            "created_at": datetime.datetime(2026, 3, 7)
        },
    ])

    db["researchers"].insert_many([
        {"username": "Sheshikala", "department": "Data Science",    "expertise": ["NLP", "Tabular"]},
        {"username": "ravi",       "department": "Computer Vision", "expertise": ["CV", "Medical Imaging"]},
        {"username": "priya",      "department": "ML Engineering",  "expertise": ["Tabular", "Deployment"]},
    ])
    print(f"  Seeded {db['experiments'].count_documents({})} experiments + {db['researchers'].count_documents({})} researchers")


# ─────────────────────────────────────────────
# PIPELINE 1: $match + $group + $sort
# ─────────────────────────────────────────────
def pipeline_by_project(db):
    print("\n── Pipeline 1: Stats by Project ────────")
    pipeline = [
        {"$match": {"status": "completed"}},          # filter early - always do this first
        {"$group": {
            "_id"          : "$project",
            "total_runs"   : {"$sum": 1},
            "avg_f1"       : {"$avg": "$metrics.test_f1"},
            "best_f1"      : {"$max": "$metrics.test_f1"},
            "total_hours"  : {"$sum": "$training_hours"}
        }},
        {"$sort": {"best_f1": -1}},
        {"$project": {
            "project"    : "$_id",
            "total_runs" : 1,
            "avg_f1"     : {"$round": ["$avg_f1", 4]},
            "best_f1"    : 1,
            "total_hours": 1,
            "_id"        : 0
        }}
    ]
    for r in db["experiments"].aggregate(pipeline):
        print(f"  {r['project']:25s} | runs: {r['total_runs']} | best_f1: {r['best_f1']} | hours: {r['total_hours']}")


# ─────────────────────────────────────────────
# PIPELINE 2: model type comparison
# ─────────────────────────────────────────────
def pipeline_model_comparison(db):
    print("\n── Pipeline 2: Model Type Comparison ───")
    pipeline = [
        {"$match": {"metrics.test_f1": {"$exists": True}}},
        {"$group": {
            "_id"    : "$model_type",
            "runs"   : {"$sum": 1},
            "avg_f1" : {"$avg": "$metrics.test_f1"},
            "avg_hrs": {"$avg": "$training_hours"}
        }},
        {"$sort": {"avg_f1": -1}},
        {"$project": {
            "model"  : "$_id",
            "runs"   : 1,
            "avg_f1" : {"$round": ["$avg_f1", 4]},
            "avg_hrs": {"$round": ["$avg_hrs", 2]},
            "_id"    : 0
        }}
    ]
    for r in db["experiments"].aggregate(pipeline):
        print(f"  {r['model']:15s} | runs: {r['runs']} | avg_f1: {r['avg_f1']} | avg_hours: {r['avg_hrs']}")


# ─────────────────────────────────────────────
# PIPELINE 3: $unwind - flatten tags array
# ─────────────────────────────────────────────
def pipeline_tag_analysis(db):
    print("\n── Pipeline 3: Tag Analysis ($unwind) ──")
    # $unwind turns ["NLP", "Transformer"] into 2 separate documents
    # so we can group and count by each tag individually
    pipeline = [
        {"$unwind": "$tags"},
        {"$group": {
            "_id"   : "$tags",
            "count" : {"$sum": 1},
            "avg_f1": {"$avg": "$metrics.test_f1"}
        }},
        {"$sort": {"count": -1}},
        {"$project": {
            "tag"   : "$_id",
            "count" : 1,
            "avg_f1": {"$round": ["$avg_f1", 4]},
            "_id"   : 0
        }}
    ]
    for r in db["experiments"].aggregate(pipeline):
        print(f"  {r['tag']:20s} | experiments: {r['count']} | avg_f1: {r['avg_f1']}")


# ─────────────────────────────────────────────
# PIPELINE 4: $lookup - JOIN researchers
# ─────────────────────────────────────────────
def pipeline_with_researcher(db):
    print("\n── Pipeline 4: Join Researchers ($lookup) ──")
    pipeline = [
        {"$lookup": {
            "from"        : "researchers",    # other collection
            "localField"  : "researcher",     # field in experiments
            "foreignField": "username",       # field in researchers
            "as"          : "researcher_info"
        }},
        {"$unwind": {"path": "$researcher_info", "preserveNullAndEmptyArrays": True}},
        {"$project": {
            "_id"       : 0,
            "experiment": "$name",
            "model"     : "$model_type",
            "researcher": "$researcher",
            "department": "$researcher_info.department",
            "test_f1"   : "$metrics.test_f1"
        }},
        {"$sort": {"test_f1": -1}},
        {"$limit": 5}
    ]
    for r in db["experiments"].aggregate(pipeline):
        print(f"  {r['experiment']:25s} | {str(r.get('department','N/A')):22s} | f1: {r.get('test_f1','N/A')}")


# ─────────────────────────────────────────────
# PIPELINE 5: $addFields - computed columns
# ─────────────────────────────────────────────
def pipeline_computed_fields(db):
    print("\n── Pipeline 5: Computed Fields ($addFields) ──")
    pipeline = [
        {"$addFields": {
            "performance_tier": {
                "$switch": {
                    "branches": [
                        {"case": {"$gte": ["$metrics.test_f1", 0.93]}, "then": "🏆 Excellent"},
                        {"case": {"$gte": ["$metrics.test_f1", 0.90]}, "then": "✅ Good"},
                        {"case": {"$gte": ["$metrics.test_f1", 0.85]}, "then": "⚠️  Average"},
                    ],
                    "default": "❌ Needs Work"
                }
            },
            "efficiency_score": {
                # F1 divided by training hours - higher means more efficient
                "$divide": ["$metrics.test_f1", "$training_hours"]
            }
        }},
        {"$project": {
            "_id"             : 0,
            "name"            : 1,
            "metrics.test_f1" : 1,
            "training_hours"  : 1,
            "performance_tier": 1,
            "efficiency_score": {"$round": ["$efficiency_score", 4]}
        }},
        {"$sort": {"efficiency_score": -1}}
    ]
    for r in db["experiments"].aggregate(pipeline):
        print(f"  {r['name']:28s} | f1: {r['metrics'].get('test_f1','N/A')} | hrs: {r['training_hours']} | eff: {r['efficiency_score']} | {r['performance_tier']}")


# ─────────────────────────────────────────────
# PIPELINE 6: $facet - multiple pipelines at once
# ─────────────────────────────────────────────
def pipeline_facet(db):
    print("\n── Pipeline 6: Multi-Facet Analytics ($facet) ──")
    pipeline = [
        {"$facet": {
            "status_breakdown": [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ],
            "project_summary": [
                {"$group": {"_id": "$project", "avg_f1": {"$avg": "$metrics.test_f1"}}},
                {"$sort": {"avg_f1": -1}},
                {"$project": {"project": "$_id", "avg_f1": {"$round": ["$avg_f1", 4]}, "_id": 0}}
            ],
            "top_3_experiments": [
                {"$match": {"status": "completed"}},
                {"$sort": {"metrics.test_f1": -1}},
                {"$limit": 3},
                {"$project": {"_id": 0, "name": 1, "test_f1": "$metrics.test_f1"}}
            ],
            "overall_stats": [
                {"$group": {
                    "_id"        : None,
                    "total"      : {"$sum": 1},
                    "avg_f1"     : {"$avg": "$metrics.test_f1"},
                    "best_f1"    : {"$max": "$metrics.test_f1"},
                    "total_hours": {"$sum": "$training_hours"}
                }}
            ]
        }}
    ]
    result = list(db["experiments"].aggregate(pipeline))[0]
    print(f"  Status    : {result['status_breakdown']}")
    print(f"  Projects  : {result['project_summary']}")
    print(f"  Top 3     : {[r['name'] for r in result['top_3_experiments']]}")
    stats = result['overall_stats'][0]
    print(f"  Overall   : total={stats['total']} | avg_f1={round(stats['avg_f1'],4)} | best_f1={stats['best_f1']} | hours={stats['total_hours']}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    db = get_db()
    seed_data(db)
    pipeline_by_project(db)
    pipeline_model_comparison(db)
    pipeline_tag_analysis(db)
    pipeline_with_researcher(db)
    pipeline_computed_fields(db)
    pipeline_facet(db)
    print("\n✅ agg_pipeline.py completed!")
