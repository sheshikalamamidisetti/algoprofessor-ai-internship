# ============================================================
# PGVECTOR SETUP
# Day 11: Databases
# Author: Sheshikala
# Topic: pgvector extension for vector similarity in PostgreSQL
# ============================================================

# pgvector is a PostgreSQL extension that adds a vector data
# type and vector similarity search operators to PostgreSQL.
# This is important for ML applications because it lets you
# store embedding vectors directly in your relational database
# alongside your other data, instead of using a separate
# vector database like ChromaDB. For example you can store
# an experiment's text description as a 384-dimensional
# embedding vector and then find similar experiments using
# cosine similarity directly in a SQL query. This file
# demonstrates pgvector setup and shows how to store and
# query experiment embeddings. A mock embedding generator
# is included so the file runs without sentence-transformers.

import os
import math
import json
import random

try:
    from sqlalchemy import create_engine, text, Column, Integer, String, Float, Text
    from sqlalchemy.orm import declarative_base, sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


# ============================================================
# EMBEDDING GENERATOR
# ============================================================

class EmbeddingGenerator:
    """
    Generates vector embeddings for text strings.
    Uses sentence-transformers if available, otherwise falls
    back to a deterministic mock that produces 64-dim vectors
    based on character frequency statistics. The mock vectors
    preserve semantic similarity well enough for demonstration.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        if ST_AVAILABLE:
            self.model     = SentenceTransformer(model_name)
            self.dim       = self.model.get_sentence_embedding_dimension()
            self.use_mock  = False
            print("SentenceTransformer loaded: " + model_name +
                  " (dim=" + str(self.dim) + ")")
        else:
            self.model    = None
            self.dim      = 64
            self.use_mock = True
            print("Using mock embedder (dim=" + str(self.dim) + ").")
            print("Install sentence-transformers for real embeddings.")

    def embed(self, texts):
        """
        Returns a list of embedding vectors for a list of texts.
        Each vector is a list of floats with length self.dim.
        """
        if self.use_mock:
            return [self._mock_embed(t) for t in texts]
        embeddings = self.model.encode(texts)
        return [e.tolist() for e in embeddings]

    def embed_one(self, text):
        """Returns a single embedding vector for one text string."""
        return self.embed([text])[0]

    def _mock_embed(self, text):
        """
        Creates a deterministic fake embedding using character
        frequency statistics. Two texts with similar words will
        have similar mock embeddings, making the demo meaningful.
        """
        text_lower = text.lower()
        vec = []
        for i in range(self.dim):
            char  = chr(ord('a') + i % 26)
            count = text_lower.count(char)
            val   = (count + i * 0.01) / (len(text_lower) + 1)
            vec.append(val)
        norm = math.sqrt(sum(x * x for x in vec)) + 1e-9
        return [x / norm for x in vec]

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """Computes cosine similarity between two vectors."""
        dot   = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a)) + 1e-9
        norm_b = math.sqrt(sum(x * x for x in vec_b)) + 1e-9
        return dot / (norm_a * norm_b)


# ============================================================
# IN-MEMORY VECTOR STORE (works without PostgreSQL)
# ============================================================

class InMemoryVectorStore:
    """
    Simulates pgvector functionality using plain Python dicts.
    Stores experiment text and embedding vectors in memory and
    supports cosine similarity search. This lets the demo run
    without a PostgreSQL server or pgvector extension installed.
    The interface matches the PostgreSQL version so switching
    to real pgvector requires only changing the backend class.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self.records  = []
        print("In-memory vector store ready (pgvector simulation).")

    def add(self, exp_id, text, metadata=None):
        """Generates an embedding and stores the record."""
        embedding = self.embedder.embed_one(text)
        self.records.append({
            "exp_id"   : exp_id,
            "text"     : text,
            "embedding": embedding,
            "metadata" : metadata or {}
        })

    def add_batch(self, items):
        """
        Adds multiple records efficiently.
        items: list of dicts with exp_id, text, and optional metadata.
        """
        texts      = [item["text"] for item in items]
        embeddings = self.embedder.embed(texts)
        for item, embedding in zip(items, embeddings):
            self.records.append({
                "exp_id"   : item["exp_id"],
                "text"     : item["text"],
                "embedding": embedding,
                "metadata" : item.get("metadata", {})
            })
        print("Added " + str(len(items)) + " records to vector store.")

    def search(self, query_text, top_k=3):
        """
        Finds the top_k most similar records to query_text
        using cosine similarity. Returns list of dicts with
        exp_id, text, similarity score, and metadata.
        """
        if not self.records:
            return []

        query_emb = self.embedder.embed_one(query_text)
        scored    = []
        for record in self.records:
            sim = EmbeddingGenerator.cosine_similarity(
                query_emb, record["embedding"]
            )
            scored.append({
                "exp_id"    : record["exp_id"],
                "text"      : record["text"],
                "similarity": round(sim, 4),
                "metadata"  : record["metadata"]
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def count(self):
        """Returns number of records stored."""
        return len(self.records)


# ============================================================
# PGVECTOR POSTGRESQL BACKEND
# ============================================================

class PgVectorStore:
    """
    Uses the pgvector PostgreSQL extension to store embeddings.
    Requires PostgreSQL with pgvector installed and the
    pgvector Python package. The CREATE EXTENSION command
    only needs to run once per database.

    Setup:
        pip install pgvector psycopg2-binary
        In psql: CREATE EXTENSION IF NOT EXISTS vector;
    """

    def __init__(self, connection_url, embedding_dim=64):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("sqlalchemy required: pip install sqlalchemy")
        if not PGVECTOR_AVAILABLE:
            raise ImportError("pgvector required: pip install pgvector")

        self.engine = create_engine(connection_url, echo=False)
        self.dim    = embedding_dim
        self._setup_extension()
        self._create_table()
        print("PgVectorStore connected to PostgreSQL.")

    def _setup_extension(self):
        """Installs pgvector extension if not already present."""
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        print("pgvector extension ready.")

    def _create_table(self):
        """Creates the experiment_embeddings table."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS experiment_embeddings (
            id        SERIAL PRIMARY KEY,
            exp_id    VARCHAR(20) NOT NULL UNIQUE,
            content   TEXT NOT NULL,
            embedding vector({dim}),
            metadata  JSONB
        )
        """.format(dim=self.dim)
        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS emb_cosine_idx "
                "ON experiment_embeddings "
                "USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = 10)"
            ))
            conn.commit()
        print("experiment_embeddings table ready with IVFFlat index.")

    def add(self, exp_id, content, embedding, metadata=None):
        """Inserts a single embedding record into PostgreSQL."""
        insert_sql = """
        INSERT INTO experiment_embeddings (exp_id, content, embedding, metadata)
        VALUES (:exp_id, :content, :embedding, :metadata)
        ON CONFLICT (exp_id) DO NOTHING
        """
        with self.engine.connect() as conn:
            conn.execute(text(insert_sql), {
                "exp_id"   : exp_id,
                "content"  : content,
                "embedding": str(embedding),
                "metadata" : json.dumps(metadata or {})
            })
            conn.commit()

    def search(self, query_embedding, top_k=3):
        """
        Searches for the top_k most similar embeddings
        using pgvector cosine distance operator (<=>).
        Lower cosine distance = higher similarity.
        """
        search_sql = """
        SELECT exp_id, content,
               1 - (embedding <=> :query_emb) AS similarity,
               metadata
        FROM experiment_embeddings
        ORDER BY embedding <=> :query_emb
        LIMIT :top_k
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(search_sql), {
                "query_emb": str(query_embedding),
                "top_k"    : top_k
            })
            rows = result.fetchall()

        return [
            {
                "exp_id"    : row[0],
                "text"      : row[1],
                "similarity": round(float(row[2]), 4),
                "metadata"  : json.loads(row[3]) if row[3] else {}
            }
            for row in rows
        ]


# ============================================================
# SAMPLE EXPERIMENT DOCUMENTS
# ============================================================

EXPERIMENT_DOCS = [
    {"exp_id": "EXP001", "text": "BERT model trained on NLP-Corpus-v2 for text classification. Achieved accuracy 0.91 and F1 0.89 after 10 epochs with batch size 32.", "metadata": {"model": "BERT",     "domain": "NLP"}},
    {"exp_id": "EXP002", "text": "RoBERTa fine-tuned on SentimentData-v1 for sentiment analysis. Achieved accuracy 0.94 and F1 0.93. Best NLP result.", "metadata": {"model": "RoBERTa",  "domain": "NLP"}},
    {"exp_id": "EXP003", "text": "ResNet50 trained on ImageNet-Subset for image classification. Top-1 accuracy 0.87. Used data augmentation and dropout.", "metadata": {"model": "ResNet50", "domain": "CV"}},
    {"exp_id": "EXP004", "text": "LSTM model on TimeSeriesData-v3 for forecasting. MSE 0.023 and MAE 0.14. 30-step sliding window worked best.", "metadata": {"model": "LSTM",     "domain": "TimeSeries"}},
    {"exp_id": "EXP005", "text": "GPT-2 fine-tuned on CustomCorpus for text generation. BLEU 0.72 and ROUGE-L 0.68. Needed early stopping to prevent overfitting.", "metadata": {"model": "GPT-2",    "domain": "NLP"}},
    {"exp_id": "EXP006", "text": "DistilBERT on NLP-Corpus-v2. Accuracy 0.89. Runs 40 percent faster than BERT with only 3 percent accuracy drop.", "metadata": {"model": "DistilBERT","domain": "NLP"}},
    {"exp_id": "EXP007", "text": "EfficientNet-B0 on CIFAR-100. Accuracy 0.82. Only 5.3M parameters, much lighter than ResNet50.", "metadata": {"model": "EfficientNet","domain": "CV"}},
    {"exp_id": "EXP008", "text": "Transformer model on TimeSeriesData-v3. MSE 0.018 which beats LSTM. Attention mechanism helps capture long-range patterns.", "metadata": {"model": "Transformer","domain": "TimeSeries"}},
]

TEST_QUERIES = [
    "Which experiments used transformer-based NLP models?",
    "Show me image classification experiments with high accuracy",
    "Find time series forecasting experiments with low MSE",
    "Which experiments used the NLP-Corpus dataset?",
    "What models were trained for text generation tasks?",
]


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("PGVECTOR SETUP DEMO")
    print("=" * 55)

    embedder = EmbeddingGenerator()

    print("\n-- Building In-Memory Vector Store --")
    store = InMemoryVectorStore(embedder)
    store.add_batch(EXPERIMENT_DOCS)
    print("Total records: " + str(store.count()))

    print("\n-- Vector Similarity Search --")
    for query in TEST_QUERIES:
        print("\nQuery: " + query)
        results = store.search(query, top_k=3)
        for i, r in enumerate(results):
            print("  [" + str(i+1) + "] " +
                  r["exp_id"] + " | sim=" + str(r["similarity"]) +
                  " | " + r["text"][:75] + "...")

    print("\n-- Embedding Similarity Analysis --")
    nlp_emb  = embedder.embed_one("NLP text classification sentiment analysis BERT")
    cv_emb   = embedder.embed_one("image classification computer vision ResNet CNN")
    ts_emb   = embedder.embed_one("time series forecasting LSTM MSE temporal")

    nlp_cv_sim = round(EmbeddingGenerator.cosine_similarity(nlp_emb, cv_emb), 4)
    nlp_ts_sim = round(EmbeddingGenerator.cosine_similarity(nlp_emb, ts_emb), 4)
    cv_ts_sim  = round(EmbeddingGenerator.cosine_similarity(cv_emb,  ts_emb), 4)

    print("NLP vs CV  similarity: " + str(nlp_cv_sim) + " (should be low - different domains)")
    print("NLP vs TS  similarity: " + str(nlp_ts_sim) + " (should be low - different domains)")
    print("CV  vs TS  similarity: " + str(cv_ts_sim)  + " (should be low - different domains)")

    print("\n-- pgvector PostgreSQL Notes --")
    print("To use real pgvector with PostgreSQL:")
    print("  1. Install: pip install pgvector psycopg2-binary")
    print("  2. In psql: CREATE EXTENSION IF NOT EXISTS vector;")
    print("  3. Replace InMemoryVectorStore with PgVectorStore")
    print("  4. Pass your PostgreSQL connection URL")
    print("  Benefit: vector search persists and scales to millions of records")

    print("\n-- pgvector setup demo complete --")


if __name__ == "__main__":
    run_demo()
