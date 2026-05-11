"""
faiss_indexer.py
----------------
Builds and queries a FAISS vector index over the DS knowledge base.
Uses sentence-transformers for embeddings (free, no API key needed).

Usage:
    python faiss_indexer.py
    python faiss_indexer.py --query "what is overfitting"
    python faiss_indexer.py --query "how to test stationarity" --top-k 3
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from knowledge_base import DS_KNOWLEDGE_BASE, get_all_texts

INDEX_PATH = "outputs/faiss_index.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"


# ── Embedding ──────────────────────────────────────────────────────────────

def get_embeddings(texts: list[str]) -> np.ndarray:
    """Generate embeddings using sentence-transformers (free, local)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        embeddings = model.encode(texts, show_progress_bar=True,
                                  normalize_embeddings=True)
        return np.array(embeddings, dtype="float32")
    except ImportError:
        print("sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        print("Falling back to random embeddings for demo...")
        np.random.seed(42)
        return np.random.rand(len(texts), 384).astype("float32")


# ── FAISS Index ────────────────────────────────────────────────────────────

def build_index(save: bool = True) -> tuple:
    """Build FAISS index from knowledge base. Returns (index, documents)."""
    try:
        import faiss
    except ImportError:
        print("faiss-cpu not installed. Run: pip install faiss-cpu")
        return None, None

    print(f"Building FAISS index from {len(DS_KNOWLEDGE_BASE)} documents...")
    texts = get_all_texts()
    embeddings = get_embeddings(texts)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine similarity (with normalised vectors)
    index.add(embeddings)
    print(f"Index built: {index.ntotal} vectors, dim={dim}")

    if save:
        Path("outputs").mkdir(exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump({"index": index, "documents": DS_KNOWLEDGE_BASE,
                         "texts": texts}, f)
        print(f"Index saved: {INDEX_PATH}")

    return index, DS_KNOWLEDGE_BASE


def load_index() -> tuple:
    """Load existing FAISS index from disk."""
    if not Path(INDEX_PATH).exists():
        print("No index found — building now...")
        return build_index()

    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"Index loaded: {data['index'].ntotal} vectors")
    return data["index"], data["documents"]


# ── Retrieval ──────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 3,
             index=None, documents=None) -> list[dict]:
    """Retrieve top-k most relevant documents for a query."""
    if index is None:
        index, documents = load_index()
    if index is None:
        return keyword_fallback(query, documents or DS_KNOWLEDGE_BASE, top_k)

    try:
        import faiss
        q_embed = get_embeddings([query])
        scores, indices = index.search(q_embed, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                doc = documents[idx].copy()
                doc["similarity_score"] = round(float(score), 4)
                results.append(doc)
        return results

    except Exception as e:
        print(f"FAISS search failed: {e} — using keyword fallback")
        return keyword_fallback(query, documents or DS_KNOWLEDGE_BASE, top_k)


def keyword_fallback(query: str, documents: list[dict],
                     top_k: int = 3) -> list[dict]:
    """Simple keyword matching fallback when FAISS is unavailable."""
    query_words = query.lower().split()
    scored = []
    for doc in documents:
        text = f"{doc['title']} {doc['content']}".lower()
        score = sum(1 for w in query_words if w in text)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        doc = doc.copy()
        doc["similarity_score"] = round(score / max(len(query_words), 1), 4)
        results.append(doc)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 7 — FAISS Indexer")
    parser.add_argument("--query",  default=None)
    parser.add_argument("--top-k",  type=int, default=3)
    parser.add_argument("--build",  action="store_true")
    args = parser.parse_args()

    if args.build or args.query is None:
        index, docs = build_index()
        print(f"\nKnowledge base indexed: {len(DS_KNOWLEDGE_BASE)} documents")
        print("Categories:")
        from collections import Counter
        cats = Counter(d["category"] for d in DS_KNOWLEDGE_BASE)
        for cat, count in cats.items():
            print(f"  {cat}: {count}")

    if args.query:
        print(f"\nQuery: {args.query}")
        print("-" * 50)
        results = retrieve(args.query, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['title']} (score: {r['similarity_score']})")
            print(f"    {r['content'][:200]}...")
