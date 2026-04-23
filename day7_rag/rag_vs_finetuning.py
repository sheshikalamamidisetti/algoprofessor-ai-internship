"""
rag_vs_finetuning.py  ·  Day 41  ·  Apr 15
--------------------------------------------
Goes into: day7_rag/  (extended from Phase 1)

Compares RAG vs fine-tuning for data science knowledge bases.
Runs the same DS queries through:
  - Plain LLM (baseline)
  - RAG with FAISS (extended from day7)
  - Fine-tuned model (from day13)
Scores each approach and explains when to use which.

Usage:
    python rag_vs_finetuning.py
    python rag_vs_finetuning.py --query "what is p-value"
    python rag_vs_finetuning.py --compare-all
"""

import argparse
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── DS Knowledge Base Documents ────────────────────────────────────────────

DS_KB = [
    {
        "id": "stat_01",
        "title": "Hypothesis Testing",
        "content": (
            "A hypothesis test evaluates two mutually exclusive statements about a population. "
            "H0 (null) states no effect exists. H1 (alternative) states an effect exists. "
            "The p-value is the probability of observing results as extreme as the data if H0 is true. "
            "If p < alpha (usually 0.05), reject H0. Type I error: rejecting true H0 (false positive). "
            "Type II error: failing to reject false H0 (false negative). "
            "Power = 1 - P(Type II error). Common tests: t-test, chi-square, ANOVA, Mann-Whitney."
        ),
    },
    {
        "id": "stat_02",
        "title": "Confidence Intervals",
        "content": (
            "A 95% CI means: if we repeated sampling 100 times, 95 of the resulting intervals "
            "would contain the true population parameter. "
            "CI = point estimate ± (critical value × standard error). "
            "For proportions: CI = p_hat ± z * sqrt(p_hat(1-p_hat)/n). "
            "Wider CI = less precision. Larger sample = narrower CI. "
            "CI and hypothesis test are equivalent: if 0 is outside the 95% CI for a difference, "
            "the difference is significant at alpha=0.05."
        ),
    },
    {
        "id": "ml_01",
        "title": "Overfitting and Regularisation",
        "content": (
            "Overfitting: model learns training data noise, fails to generalise. "
            "Symptoms: train accuracy >> test accuracy (gap > 15% is concerning). "
            "Causes: model too complex, insufficient data, no regularisation. "
            "Solutions: L1 regularisation (Lasso) shrinks coefficients to zero — feature selection. "
            "L2 regularisation (Ridge) shrinks all coefficients — reduces magnitude. "
            "Dropout: randomly deactivates neurons during training. "
            "Cross-validation: k-fold CV gives unbiased generalisation estimate. "
            "Early stopping: halt training when validation loss stops improving."
        ),
    },
    {
        "id": "ts_01",
        "title": "Time Series Stationarity",
        "content": (
            "A stationary time series has constant mean, variance, and autocorrelation over time. "
            "ARIMA requires stationarity. Test with Augmented Dickey-Fuller (ADF): "
            "if p < 0.05, series is stationary. "
            "To make stationary: differencing (subtract lag-1 value), "
            "log transform (stabilises variance), seasonal differencing. "
            "ACF plot shows autocorrelation at each lag. "
            "PACF shows partial autocorrelation (controlling for intermediate lags). "
            "AR(p): PACF cuts off at lag p. MA(q): ACF cuts off at lag q."
        ),
    },
    {
        "id": "ml_02",
        "title": "Feature Importance and Selection",
        "content": (
            "Random Forest feature importance: mean decrease in impurity across all trees. "
            "Limitation: biased toward high-cardinality features. "
            "Permutation importance: more reliable — measures drop in score when feature is shuffled. "
            "SHAP values: game-theoretic, explains individual predictions. "
            "Filter methods: correlation, mutual information — fast, model-independent. "
            "Wrapper methods: RFE (Recursive Feature Elimination) — uses model performance. "
            "For time series: ensure no future leakage — only use features available at prediction time."
        ),
    },
]

# ── RAG Pipeline ───────────────────────────────────────────────────────────

def build_faiss_index(documents: list[dict]) -> tuple:
    """Build FAISS index from documents."""
    import numpy as np
    try:
        import faiss
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        texts = [f"{d['title']}: {d['content']}" for d in documents]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        embeddings = np.array([r.embedding for r in response.data], dtype="float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatCosine(dim)
        index.add(embeddings)
        print(f"FAISS index built: {len(documents)} docs, dim={dim}")
        return index, embeddings, client

    except ImportError:
        print("faiss-cpu not installed — using simple cosine similarity fallback")
        return None, None, None


def retrieve(query: str, index, embeddings, documents: list[dict],
             client, top_k: int = 2) -> list[dict]:
    """Retrieve top-k relevant documents for a query."""
    import numpy as np

    if index is None:
        # Simple keyword fallback
        query_lower = query.lower()
        scored = []
        for doc in documents:
            score = sum(1 for w in query_lower.split()
                       if w in doc["content"].lower())
            scored.append((score, doc))
        scored.sort(reverse=True)
        return [d for _, d in scored[:top_k]]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    )
    q_embed = np.array([response.data[0].embedding], dtype="float32")
    _, indices = index.search(q_embed, top_k)
    return [documents[i] for i in indices[0]]


def rag_answer(query: str, index, embeddings, documents, client) -> dict:
    """Answer using RAG — retrieve then generate."""
    from openai import OpenAI
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    retrieved = retrieve(query, index, embeddings, documents, client)
    context = "\n\n".join([f"[{d['title']}]\n{d['content']}" for d in retrieved])

    t0 = time.time()
    resp = llm.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content":
             "You are a data science expert. Answer using only the provided context."},
            {"role": "user", "content":
             f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        max_tokens=512,
    )
    return {
        "method": "RAG",
        "answer": resp.choices[0].message.content,
        "sources": [d["title"] for d in retrieved],
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


def plain_llm_answer(query: str) -> dict:
    """Answer using plain LLM — no retrieval."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    t0 = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data science expert."},
            {"role": "user", "content": query},
        ],
        max_tokens=512,
    )
    return {
        "method": "Plain LLM",
        "answer": resp.choices[0].message.content,
        "sources": [],
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


def compare_approaches(query: str):
    """Compare plain LLM vs RAG on the same query."""
    print(f"\nQuery: {query}")
    print("=" * 70)

    index, embeddings, client = build_faiss_index(DS_KB)

    plain = plain_llm_answer(query)
    rag   = rag_answer(query, index, embeddings, DS_KB, client)

    for result in [plain, rag]:
        print(f"\n── {result['method']} ({result['latency_ms']}ms) ──")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")
        print(result["answer"][:500])

    print("\n── When to use RAG vs Fine-tuning ──")
    print("RAG:          when knowledge updates frequently, source attribution needed")
    print("Fine-tuning:  when style/behaviour change needed, not just new facts")
    print("Both:         RAG for facts + fine-tuning for analyst communication style")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 41 — RAG vs Fine-tuning")
    parser.add_argument("--query", default="What is overfitting and how do I fix it?")
    parser.add_argument("--compare-all", action="store_true")
    args = parser.parse_args()

    if args.compare_all:
        queries = [
            "What is overfitting and how do I fix it?",
            "How do I test if my time series is stationary?",
            "What is the difference between Type I and Type II error?",
        ]
        for q in queries:
            compare_approaches(q)
    else:
        compare_approaches(args.query)
