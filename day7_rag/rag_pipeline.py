"""
rag_pipeline.py
---------------
Complete RAG pipeline for the DS/Stats/ML knowledge base.
Retrieve relevant documents from FAISS, then generate
an answer using an LLM grounded in that context.

Usage:
    python rag_pipeline.py
    python rag_pipeline.py --query "what is a p-value"
    python rag_pipeline.py --query "how do I fix overfitting" --model gpt4o
    python rag_pipeline.py --demo    # runs without any API key
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from knowledge_base import DS_KNOWLEDGE_BASE
from faiss_indexer import retrieve

load_dotenv()

# ── Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(query: str, retrieved_docs: list[dict]) -> str:
    context = "\n\n".join([
        f"[{i+1}] {doc['title']}\n{doc['content']}"
        for i, doc in enumerate(retrieved_docs)
    ])
    return f"""You are a data science expert. Answer the question using ONLY
the provided context. If the answer is not in the context, say so.
Cite which source number supports your answer.

Context:
{context}

Question: {query}

Answer:"""


# ── LLM Generation ────────────────────────────────────────────────────────

def generate_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data science expert."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
    )
    return resp.choices[0].message.content


def generate_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def generate_demo(prompt: str, query: str,
                  docs: list[dict]) -> str:
    """Generate a demo answer without API — uses retrieved docs directly."""
    titles = [d["title"] for d in docs]
    content_snippet = docs[0]["content"][:300] if docs else ""
    return (
        f"Based on the retrieved documents ({', '.join(titles)}), "
        f"here is the answer to '{query}':\n\n"
        f"{content_snippet}...\n\n"
        f"[Sources: {', '.join(f'[{i+1}]' for i in range(len(docs)))}]\n"
        f"[Demo mode — add OPENAI_API_KEY for full LLM generation]"
    )


# ── Full RAG pipeline ──────────────────────────────────────────────────────

def rag_query(query: str, top_k: int = 3,
              model: str = "demo") -> dict:
    """Run the complete RAG pipeline on a query."""
    import time
    t0 = time.time()

    # Step 1 — Retrieve
    docs = retrieve(query, top_k=top_k)

    # Step 2 — Build prompt
    prompt = build_prompt(query, docs)

    # Step 3 — Generate
    if model == "gpt4o" and os.getenv("OPENAI_API_KEY"):
        answer = generate_openai(prompt)
    elif model == "claude" and os.getenv("ANTHROPIC_API_KEY"):
        answer = generate_claude(prompt)
    else:
        answer = generate_demo(prompt, query, docs)

    latency = round((time.time() - t0) * 1000, 1)

    return {
        "query":       query,
        "model":       model,
        "retrieved":   [{"id": d["id"], "title": d["title"],
                          "score": d.get("similarity_score", 0)} for d in docs],
        "answer":      answer,
        "latency_ms":  latency,
        "timestamp":   datetime.now().isoformat(),
    }


def run_demo():
    """Run demo queries and save results."""
    print("=" * 60)
    print("Day 7 — RAG Pipeline Demo")
    print("Knowledge base: Statistics, ML, Time Series, Deep Learning")
    print("=" * 60)

    queries = [
        "What is overfitting and how do I fix it?",
        "How do I test if my time series is stationary?",
        "What is the difference between Type I and Type II error?",
        "Which feature importance method should I use for Random Forest?",
        "How do I choose between ARIMA and seasonal ARIMA?",
    ]

    results = []
    for query in queries:
        print(f"\nQuery: {query}")
        result = rag_query(query, model="demo")

        print(f"Retrieved: {[r['title'] for r in result['retrieved']]}")
        print(f"Answer:    {result['answer'][:300]}...")
        print(f"Latency:   {result['latency_ms']}ms")
        results.append(result)

    # Save
    Path("outputs").mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/rag_results_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 7 — RAG Pipeline")
    parser.add_argument("--query", default=None)
    parser.add_argument("--model", default="demo",
                        choices=["demo", "gpt4o", "claude"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--demo",  action="store_true")
    args = parser.parse_args()

    if args.query:
        result = rag_query(args.query, top_k=args.top_k, model=args.model)
        print(f"\nQuery:     {result['query']}")
        print(f"Retrieved: {[r['title'] for r in result['retrieved']]}")
        print(f"\nAnswer:\n{result['answer']}")
    else:
        run_demo()
