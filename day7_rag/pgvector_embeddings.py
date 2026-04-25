"""
pgvector_embeddings.py  ·  Day 43  ·  Apr 17
----------------------------------------------
Goes into: day7_rag/  (extended, leverages day11_databases pgvector)

Stores statistical report embeddings in PostgreSQL via pgvector.
Semantic search over DS reports — extends day11_databases pgvector work.

Usage:
    python pgvector_embeddings.py --setup     # create table
    python pgvector_embeddings.py --index     # embed + store reports
    python pgvector_embeddings.py --search "find reports about churn"
    python pgvector_embeddings.py --demo      # runs without Postgres
"""

import argparse
import json
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/llm_engineering")

# ── Sample statistical reports to embed ───────────────────────────────────

REPORTS = [
    {
        "id": 1,
        "title": "Q1 2024 Churn Analysis Report",
        "content": (
            "Customer churn rate increased to 8.2% in Q1 2024, up from 6.5% in Q4 2023. "
            "Logistic regression identified tenure < 12 months and monthly charges > $80 "
            "as the top predictors. ANOVA confirmed significant churn rate differences "
            "across contract types (F=12.4, p<0.001). Recommended intervention: "
            "loyalty discount for 6-12 month customers with high monthly charges."
        ),
        "report_type": "churn",
        "date": "2024-04-01",
    },
    {
        "id": 2,
        "title": "Revenue Forecasting — ARIMA vs Prophet Comparison",
        "content": (
            "Compared ARIMA(2,1,1) and Prophet for monthly revenue forecasting. "
            "ADF test confirmed stationarity after first differencing (p=0.003). "
            "ARIMA RMSE: 12,400. Prophet RMSE: 10,200. Prophet outperforms on "
            "datasets with strong seasonality. 12-month forecast shows 15% YoY growth. "
            "95% prediction intervals widen significantly after 6 months."
        ),
        "report_type": "forecasting",
        "date": "2024-03-15",
    },
    {
        "id": 3,
        "title": "A/B Test Report — Pricing Page Redesign",
        "content": (
            "Ran independent samples t-test on conversion rates: control=3.2%, variant=4.1%. "
            "t=3.41, p=0.0007, reject H0 at alpha=0.05. Cohen's d=0.34 (small-medium effect). "
            "95% CI for difference: [0.4%, 1.6%]. Statistical power=0.89. "
            "Recommendation: roll out new pricing page. Expected annual revenue lift: $240k."
        ),
        "report_type": "ab_test",
        "date": "2024-02-28",
    },
    {
        "id": 4,
        "title": "ML Model Performance Dashboard — March 2024",
        "content": (
            "Random Forest churn model: train accuracy=0.94, test accuracy=0.79 — "
            "moderate overfitting detected (gap=15%). Applied L2 regularisation, "
            "new test accuracy=0.83. Top features: tenure (SHAP=0.31), "
            "monthly_charges (SHAP=0.28), contract_type (SHAP=0.19). "
            "ROC-AUC=0.87. Recommend retraining quarterly with fresh data."
        ),
        "report_type": "model_performance",
        "date": "2024-03-31",
    },
]


# ── PostgreSQL + pgvector setup ────────────────────────────────────────────

SETUP_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ds_reports (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    report_type VARCHAR(50),
    report_date DATE,
    embedding   VECTOR(1536),
    metadata    JSONB,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ds_reports_embedding_idx
    ON ds_reports USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
"""


def setup_db(conn):
    with conn.cursor() as cur:
        cur.execute(SETUP_SQL)
    conn.commit()
    print("pgvector table and index created")


def embed_text(text: str, client) -> list[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def index_reports(conn, client):
    with conn.cursor() as cur:
        for report in REPORTS:
            text = f"{report['title']}. {report['content']}"
            embedding = embed_text(text, client)
            cur.execute(
                """INSERT INTO ds_reports
                   (title, content, report_type, report_date, embedding, metadata)
                   VALUES (%s, %s, %s, %s, %s::vector, %s)
                   ON CONFLICT DO NOTHING""",
                (
                    report["title"],
                    report["content"],
                    report["report_type"],
                    report["date"],
                    str(embedding),
                    json.dumps({"id": report["id"]}),
                ),
            )
            print(f"  Indexed: {report['title'][:50]}")
    conn.commit()
    print(f"\nIndexed {len(REPORTS)} reports into pgvector")


def semantic_search(query: str, conn, client, top_k: int = 3) -> list[dict]:
    q_embed = embed_text(query, client)
    with conn.cursor() as cur:
        cur.execute(
            """SELECT title, content, report_type,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM ds_reports
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (str(q_embed), str(q_embed), top_k),
        )
        rows = cur.fetchall()
    return [
        {"title": r[0], "content": r[1][:200],
         "type": r[2], "similarity": round(float(r[3]), 4)}
        for r in rows
    ]


def run_demo():
    """Demo without Postgres — shows what the output looks like."""
    print("=" * 60)
    print("Day 43 — pgvector Semantic Search Demo")
    print("=" * 60)
    print()

    queries = [
        "customer churn prediction",
        "time series forecasting ARIMA",
        "A/B test statistical significance",
        "machine learning model overfitting",
    ]

    # Simulate results
    demo_results = {
        "customer churn prediction": [
            {"title": "Q1 2024 Churn Analysis Report",
             "similarity": 0.91, "type": "churn"},
            {"title": "ML Model Performance Dashboard — March 2024",
             "similarity": 0.78, "type": "model_performance"},
        ],
        "time series forecasting ARIMA": [
            {"title": "Revenue Forecasting — ARIMA vs Prophet Comparison",
             "similarity": 0.95, "type": "forecasting"},
        ],
        "A/B test statistical significance": [
            {"title": "A/B Test Report — Pricing Page Redesign",
             "similarity": 0.93, "type": "ab_test"},
        ],
        "machine learning model overfitting": [
            {"title": "ML Model Performance Dashboard — March 2024",
             "similarity": 0.89, "type": "model_performance"},
        ],
    }

    for query in queries:
        print(f"Query: '{query}'")
        for r in demo_results.get(query, []):
            bar = "█" * int(r["similarity"] * 20)
            print(f"  [{r['similarity']:.2f}] {bar} {r['title']}")
        print()

    print("Why pgvector over FAISS for production:")
    print("  FAISS:    in-memory, fast, no persistence, no SQL joins")
    print("  pgvector: persistent, supports SQL filters, joins with other tables,")
    print("            works with existing PostgreSQL infra from day11_databases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 43 — pgvector Embeddings")
    parser.add_argument("--setup",  action="store_true")
    parser.add_argument("--index",  action="store_true")
    parser.add_argument("--search", default=None)
    parser.add_argument("--demo",   action="store_true", default=True)
    args = parser.parse_args()

    if args.demo and not (args.setup or args.index or args.search):
        run_demo()
        exit(0)

    from openai import OpenAI
    import psycopg2
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    conn = psycopg2.connect(DB_URL)

    if args.setup:
        setup_db(conn)
    if args.index:
        index_reports(conn, client)
    if args.search:
        results = semantic_search(args.search, conn, client)
        print(f"\nResults for: '{args.search}'")
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['title']} ({r['type']})")
            print(f"           {r['content'][:100]}...")

    conn.close()
