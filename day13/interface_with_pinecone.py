"""
Day 13: Pinecone Cloud Vector Database
Author: Sheshikala
Date: 2026-03-13
"""

import os
import time
import numpy as np


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_API_KEY_HERE")
INDEX_NAME = "ml-experiments"
DIMENSION = 384


EXPERIMENT_DOCS = [
    {"id": "exp_001", "text": "BERT fine-tuned for sentiment analysis achieving 90.2 F1 score"},
    {"id": "exp_002", "text": "RoBERTa outperforms BERT on sentiment classification with 93.6 F1"},
    {"id": "exp_003", "text": "DistilBERT lightweight model for production deployment"},
    {"id": "exp_004", "text": "ResNet50 transfer learning for medical chest X-ray classification"},
    {"id": "exp_005", "text": "XGBoost gradient boosting for fraud detection"},
    {"id": "exp_006", "text": "LightGBM gradient boosting for sales forecasting"},
]


def get_embeddings(texts):

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        return model.encode(texts, show_progress_bar=False)

    except ImportError:

        print("sentence-transformers not installed - using random vectors")

        np.random.seed(42)

        return np.random.randn(len(texts), DIMENSION).astype("float32")


def run_pinecone():

    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        print("pinecone-client not installed. Run: pip install pinecone-client")
        run_demo_mode()
        return

    if PINECONE_API_KEY == "YOUR_API_KEY_HERE":

        print("No Pinecone API key found - running in demo mode")

        run_demo_mode()

        return

    print("Connecting to Pinecone")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    print("\nCreating index")

    existing = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing:

        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        print("Index created:", INDEX_NAME)

        time.sleep(3)

    else:

        print("Index already exists:", INDEX_NAME)

    index = pc.Index(INDEX_NAME)

    print("\nUpserting vectors")

    texts = [doc["text"] for doc in EXPERIMENT_DOCS]

    embeddings = get_embeddings(texts)

    vectors = [
        (
            doc["id"],
            emb.tolist(),
            {"text": doc["text"]},
        )
        for doc, emb in zip(EXPERIMENT_DOCS, embeddings)
    ]

    index.upsert(vectors=vectors)

    print("Vectors inserted:", len(vectors))

    time.sleep(2)

    print("\nQuery example")

    query_text = "transformer model for NLP classification"

    query_emb = get_embeddings([query_text])[0]

    results = index.query(
        vector=query_emb.tolist(),
        top_k=3,
        include_metadata=True,
    )

    print("Query:", query_text)

    for match in results["matches"]:

        print(match["id"], "score:", round(match["score"], 4), "|", match["metadata"]["text"][:60])

    print("\nIndex statistics")

    stats = index.describe_index_stats()

    print("Total vectors:", stats["total_vector_count"])

    print("Dimension:", stats["dimension"])

    print("\nDelete example")

    index.delete(ids=["exp_001"])

    print("Deleted vector exp_001")

    time.sleep(1)

    stats = index.describe_index_stats()

    print("Remaining vectors:", stats["total_vector_count"])


def run_demo_mode():

    print("=" * 50)
    print("PINECONE DEMO MODE")
    print("=" * 50)

    print("\nThis shows what would happen with a real API key\n")

    print("Step 1: Connect to Pinecone")

    print("pc = Pinecone(api_key='your_key')")

    print("\nStep 2: Create index")

    print("pc.create_index(")
    print("  name =", INDEX_NAME)
    print("  dimension =", DIMENSION)
    print("  metric = cosine")
    print("  cloud = aws region = us-east-1")
    print(")")

    print("\nStep 3: Generate embeddings")

    texts = [doc["text"] for doc in EXPERIMENT_DOCS]

    embeddings = get_embeddings(texts)

    print("Encoded texts:", len(texts))

    print("\nStep 4: Upsert vectors")

    print("Vectors would be inserted:", len(EXPERIMENT_DOCS))

    print("\nStep 5: Query example")

    query = "transformer model for sentiment"

    query_emb = get_embeddings([query])[0]

    similarities = []

    for i, emb in enumerate(embeddings):

        score = float(
            np.dot(query_emb, emb) /
            (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        )

        similarities.append((EXPERIMENT_DOCS[i]["id"], score, EXPERIMENT_DOCS[i]["text"]))

    similarities.sort(key=lambda x: x[1], reverse=True)

    for doc_id, score, text in similarities[:3]:

        print(doc_id, "score:", round(score, 4), "|", text[:60])

    print("\nTo use real Pinecone:")
    print("1. Sign up at https://app.pinecone.io")
    print("2. Create API key")
    print("3. Set PINECONE_API_KEY in environment")


if __name__ == "__main__":

    print("Day 13: Pinecone Vector Database")

    print("=" * 40)

    run_pinecone()

    print("\ninterface_with_pinecone.py completed")
