# ============================================================
# ADVANCED RAG TECHNIQUES
# Day 15: Graph RAG + Advanced Retrieval
# Author: Sheshikala
# Topic: HyDE, query expansion, reranking, multi-query RAG
# ============================================================

# Basic RAG just embeds the query and searches.
# Advanced RAG improves retrieval quality by transforming
# the query before searching, or searching multiple times
# and combining results. I learned these techniques from
# research papers and found them very effective.

import math
import re
from typing import List, Dict


# ============================================================
# SHARED KNOWLEDGE BASE
# ============================================================

KNOWLEDGE_BASE = [
    {"id": "K01", "text": "Experiment EXP001: Ananya trained BERT on NLP-Corpus-v2. Accuracy 0.91, F1 0.89. Used 10 epochs, batch size 32, learning rate 2e-5."},
    {"id": "K02", "text": "Experiment EXP002: Vikram fine-tuned RoBERTa on SentimentData-v1. Accuracy 0.94, F1 0.93. Best NLP result in the project."},
    {"id": "K03", "text": "Experiment EXP003: Priya trained ResNet50 on ImageNet-Subset. Top-1 accuracy 0.87, Top-5 accuracy 0.96. Used data augmentation."},
    {"id": "K04", "text": "Experiment EXP004: Rohan ran LSTM on TimeSeriesData-v3. MSE 0.023, MAE 0.14. 30-step window worked best."},
    {"id": "K05", "text": "Experiment EXP005: Ananya fine-tuned GPT-2 on CustomCorpus. BLEU 0.72, ROUGE-L 0.68. Needed early stopping to prevent overfitting."},
    {"id": "K06", "text": "Experiment EXP006: Vikram tested DistilBERT on NLP-Corpus-v2. Accuracy 0.89, inference time 12ms. 40% faster than BERT."},
    {"id": "K07", "text": "Experiment EXP007: Priya trained EfficientNet-B0 on CIFAR-100. Accuracy 0.82, only 5.3M parameters. Much lighter than ResNet50."},
    {"id": "K08", "text": "Experiment EXP008: Rohan ran Transformer on TimeSeriesData-v3. MSE 0.018, better than LSTM. Attention captures long-range dependencies."},
    {"id": "K09", "text": "Project NLP-Research led by Ananya. Uses NLP-Corpus-v2, SentimentData-v1, CustomCorpus. Focus: text classification and language generation."},
    {"id": "K10", "text": "Project CV-Research led by Priya. Uses ImageNet-Subset and CIFAR-100. Focus: image recognition with CNNs."},
    {"id": "K11", "text": "Dataset NLP-Corpus-v2: 500k news documents, 80/10/10 split, tokenized with BERT tokenizer."},
    {"id": "K12", "text": "Dataset TimeSeriesData-v3: 200k time series samples, hourly frequency, normalized to zero mean unit variance."},
]


# ============================================================
# SIMPLE EMBEDDER (no external dependencies)
# ============================================================

def simple_embed(text, dim=64):
    """Deterministic fake embedding using character statistics."""
    text = text.lower()
    vec = []
    for i in range(dim):
        char = chr(ord('a') + i % 26)
        count = text.count(char)
        vec.append((count + i * 0.01) / (len(text) + 1))
    norm = math.sqrt(sum(x**2 for x in vec)) + 1e-9
    return [x / norm for x in vec]

def cosine_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a)) + 1e-9
    nb = math.sqrt(sum(x**2 for x in b)) + 1e-9
    return dot / (na * nb)

def retrieve(query, docs, top_k=3):
    """Basic semantic retrieval over knowledge base."""
    q_emb = simple_embed(query)
    scored = []
    for doc in docs:
        d_emb = simple_embed(doc["text"])
        score = cosine_sim(q_emb, d_emb)
        scored.append({"id": doc["id"], "text": doc["text"], "score": round(score, 4)})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


# ============================================================
# TECHNIQUE 1: HyDE (Hypothetical Document Embedding)
# ============================================================

def generate_hypothetical_answer(query):
    """
    Generates a fake 'ideal answer' for the query.
    We then embed this hypothetical answer instead of the raw query.
    The idea: an ideal answer is more similar to relevant docs
    than the raw question is.

    In production, this uses an LLM. Here we use templates.
    I found HyDE helps a lot for vague or short queries.
    """
    query_lower = query.lower()

    if "accuracy" in query_lower or "performance" in query_lower:
        return (
            "The experiment achieved high accuracy. The model was trained "
            "for multiple epochs and evaluated on a held-out test set. "
            "Accuracy and F1 score were recorded as the main metrics."
        )
    elif "researcher" in query_lower or "who" in query_lower:
        return (
            "The researcher ran this experiment as part of the project. "
            "They configured the model, selected the dataset, and analyzed results."
        )
    elif "dataset" in query_lower or "data" in query_lower:
        return (
            "The dataset contains thousands of samples split into train, validation, "
            "and test sets. It was preprocessed and tokenized before use."
        )
    elif "model" in query_lower or "architecture" in query_lower:
        return (
            "The model is a deep learning architecture trained end-to-end. "
            "It uses transformer layers or convolutional layers depending on the task."
        )
    elif "time series" in query_lower or "forecast" in query_lower:
        return (
            "The time series experiment used LSTM or Transformer model. "
            "MSE and MAE were the evaluation metrics. A sliding window approach was used."
        )
    else:
        return f"This is an experiment record about {query} in the ML tracker system."


class HyDERetriever:
    """
    HyDE: instead of embedding the raw query, we generate a
    hypothetical answer and embed that. Retrieves more relevant docs
    for complex or abstract queries.
    """
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=3):
        hypothetical = generate_hypothetical_answer(query)
        combined = query + " " + hypothetical
        results = retrieve(combined, self.docs, top_k)
        return {"hypothetical": hypothetical, "results": results}


# ============================================================
# TECHNIQUE 2: QUERY EXPANSION
# ============================================================

SYNONYM_MAP = {
    "accuracy": ["accuracy", "performance", "score", "result"],
    "researcher": ["researcher", "scientist", "author", "who ran"],
    "dataset": ["dataset", "data", "corpus", "collection"],
    "model": ["model", "architecture", "network", "algorithm"],
    "experiment": ["experiment", "trial", "run", "test"],
    "nlp": ["nlp", "natural language", "text", "language model"],
    "cv": ["cv", "computer vision", "image", "visual"],
    "bert": ["bert", "transformer", "language model"],
    "lstm": ["lstm", "recurrent", "rnn", "sequence model"],
    "best": ["best", "highest", "top", "maximum", "optimal"],
}

def expand_query(query):
    """
    Adds synonyms and related terms to the query.
    Improves recall by catching more relevant documents
    even when they use different terminology.
    """
    query_lower = query.lower()
    expansions = set()
    for keyword, synonyms in SYNONYM_MAP.items():
        if keyword in query_lower:
            expansions.update(synonyms)
    if expansions:
        extra = " ".join(expansions - set(query_lower.split()))
        return query + " " + extra
    return query


class QueryExpansionRetriever:
    """
    Expands the query with synonyms before retrieval.
    Helps when user uses informal or abbreviated language.
    """
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=3):
        expanded = expand_query(query)
        results = retrieve(expanded, self.docs, top_k)
        return {"expanded_query": expanded, "results": results}


# ============================================================
# TECHNIQUE 3: CROSS-ENCODER RERANKING
# ============================================================

def cross_encoder_score(query, doc_text):
    """
    Simulates cross-encoder relevance scoring.
    Real cross-encoders process query+doc together for better accuracy.
    Here we use keyword overlap as a proxy.
    """
    query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
    doc_words = set(re.sub(r'[^\w\s]', '', doc_text.lower()).split())
    if not query_words:
        return 0.0
    overlap = len(query_words & doc_words)
    # bonus for exact phrase match
    phrase_bonus = 0.2 if any(w in doc_text.lower() for w in query_words if len(w) > 4) else 0.0
    return (overlap / len(query_words)) + phrase_bonus


class RerankingRetriever:
    """
    First retrieves a larger candidate set, then reranks using
    cross-encoder scores. More accurate than single-stage retrieval
    but slower because it scores each candidate individually.
    """
    def __init__(self, docs, initial_k=8):
        self.docs = docs
        self.initial_k = initial_k

    def retrieve(self, query, top_k=3):
        # stage 1: bi-encoder retrieval (fast, gets candidates)
        candidates = retrieve(query, self.docs, top_k=self.initial_k)
        # stage 2: cross-encoder reranking (accurate, smaller set)
        for c in candidates:
            c["rerank_score"] = cross_encoder_score(query, c["text"])
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# ============================================================
# TECHNIQUE 4: MULTI-QUERY RAG
# ============================================================

def generate_sub_queries(query):
    """
    Breaks a complex query into simpler sub-queries.
    Each sub-query targets a different aspect of the question.
    I use templates here; production would use an LLM.
    """
    query_lower = query.lower()
    sub_queries = [query]  # always include original

    if "best" in query_lower or "highest" in query_lower:
        sub_queries.append("which experiment achieved highest metric score")
        sub_queries.append("top performing model in experiments")

    if "researcher" in query_lower or "who" in query_lower:
        sub_queries.append("researcher names and their experiments")
        sub_queries.append("which scientist ran which model")

    if "compare" in query_lower or "vs" in query_lower:
        sub_queries.append("experiment results comparison")
        sub_queries.append("model performance differences")

    if "dataset" in query_lower:
        sub_queries.append("datasets used in experiments")
        sub_queries.append("dataset properties and size")

    return list(set(sub_queries))[:4]   # max 4 sub-queries


def reciprocal_rank_fusion(all_results, k=60):
    """Merges results from multiple queries using RRF."""
    scores = {}
    text_map = {}
    for results in all_results:
        for rank, doc in enumerate(results):
            text = doc["text"]
            scores[text] = scores.get(text, 0) + 1 / (rank + 1 + k)
            text_map[text] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"text": t, "rrf_score": round(s, 6),
             "id": text_map[t].get("id", "")} for t, s in ranked]


class MultiQueryRetriever:
    """
    Generates multiple sub-queries, retrieves for each,
    then merges results using RRF. Improves recall for
    complex questions that have multiple aspects.
    """
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=3):
        sub_queries = generate_sub_queries(query)
        all_results = []
        for sq in sub_queries:
            results = retrieve(sq, self.docs, top_k=top_k)
            all_results.append(results)
        fused = reciprocal_rank_fusion(all_results)
        return {"sub_queries": sub_queries, "results": fused[:top_k]}


# ============================================================
# TECHNIQUE 5: CONTEXTUAL COMPRESSION
# ============================================================

def compress_chunk(query, chunk_text, max_sentences=2):
    """
    Extracts only the most relevant sentences from a chunk.
    Reduces context length sent to LLM while keeping relevance.
    I learned this technique helps when chunks are long and noisy.
    """
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
    query_words = set(query.lower().split())

    scored_sents = []
    for sent in sentences:
        overlap = sum(1 for w in sent.lower().split() if w in query_words)
        scored_sents.append((overlap, sent))

    scored_sents.sort(reverse=True)
    top_sents = [s for _, s in scored_sents[:max_sentences]]
    return " ".join(top_sents)


class ContextualCompressionRetriever:
    """
    Retrieves documents then compresses each chunk to keep
    only the most query-relevant sentences.
    Reduces noise and token count for the LLM.
    """
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, top_k=3):
        results = retrieve(query, self.docs, top_k=top_k)
        for r in results:
            r["compressed"] = compress_chunk(query, r["text"])
        return results


# ============================================================
# TECHNIQUE COMPARISON
# ============================================================

def compare_techniques(query, docs, top_k=3):
    """Runs all 5 techniques and shows results side by side."""
    print(f"\nQuery: {query}")
    print("-" * 50)

    # baseline
    baseline = retrieve(query, docs, top_k)
    print(f"\n[Baseline] Top result: {baseline[0]['text'][:80]}...")

    # HyDE
    hyde = HyDERetriever(docs)
    hyde_result = hyde.retrieve(query, top_k)
    print(f"[HyDE]     Hypothetical: {hyde_result['hypothetical'][:60]}...")
    print(f"           Top result:   {hyde_result['results'][0]['text'][:80]}...")

    # Query Expansion
    qe = QueryExpansionRetriever(docs)
    qe_result = qe.retrieve(query, top_k)
    print(f"[QueryExp] Expanded: {qe_result['expanded_query'][:60]}...")
    print(f"           Top result: {qe_result['results'][0]['text'][:80]}...")

    # Reranking
    rr = RerankingRetriever(docs)
    rr_result = rr.retrieve(query, top_k)
    print(f"[Rerank]   Top result: {rr_result[0]['text'][:80]}...")
    print(f"           Rerank score: {rr_result[0]['rerank_score']:.4f}")

    # Multi-query
    mq = MultiQueryRetriever(docs)
    mq_result = mq.retrieve(query, top_k)
    print(f"[MultiQ]   Sub-queries: {mq_result['sub_queries']}")
    print(f"           Top result: {mq_result['results'][0]['text'][:80]}...")

    # Compression
    cc = ContextualCompressionRetriever(docs)
    cc_result = cc.retrieve(query, top_k)
    print(f"[Compress] Compressed: {cc_result[0]['compressed'][:80]}...")


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("ADVANCED RAG TECHNIQUES DEMO")
    print("=" * 55)

    test_queries = [
        "Which researcher achieved the best accuracy?",
        "What datasets were used in NLP experiments?",
        "Compare LSTM and Transformer for time series",
        "Who ran experiments with transformer models?",
    ]

    for q in test_queries:
        compare_techniques(q, KNOWLEDGE_BASE, top_k=3)

    print("\n-- Advanced RAG demo complete --")


if __name__ == "__main__":
    run_demo()
