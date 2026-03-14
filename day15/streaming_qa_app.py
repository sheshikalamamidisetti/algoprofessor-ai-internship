# ============================================================
# STREAMING QA APP
# Day 15: Graph RAG + Advanced Retrieval
# Author: Sheshikala
# Topic: Streaming responses and multi-turn conversation
# ============================================================

# Regular RAG waits for the full answer before showing anything.
# Streaming sends tokens as they are generated, like ChatGPT.
# I also added multi-turn support so the app remembers
# previous questions in the same session.

import time
import math
import re
from datetime import datetime
from typing import List, Dict, Optional


# ============================================================
# KNOWLEDGE BASE
# ============================================================

QA_KNOWLEDGE_BASE = [
    {"id": "Q01", "text": "EXP001 by Ananya used BERT on NLP-Corpus-v2. Accuracy 0.91, F1 0.89. Ran 10 epochs batch size 32."},
    {"id": "Q02", "text": "EXP002 by Vikram used RoBERTa on SentimentData-v1. Accuracy 0.94, F1 0.93. Best NLP result."},
    {"id": "Q03", "text": "EXP003 by Priya used ResNet50 on ImageNet-Subset. Top-1 accuracy 0.87. Used data augmentation."},
    {"id": "Q04", "text": "EXP004 by Rohan used LSTM on TimeSeriesData-v3. MSE 0.023, MAE 0.14. 30-step window."},
    {"id": "Q05", "text": "EXP005 by Ananya used GPT-2 on CustomCorpus. BLEU 0.72. Needed early stopping."},
    {"id": "Q06", "text": "EXP006 by Vikram used DistilBERT on NLP-Corpus-v2. Accuracy 0.89. 40% faster than BERT."},
    {"id": "Q07", "text": "EXP007 by Priya used EfficientNet-B0 on CIFAR-100. Accuracy 0.82. Only 5.3M parameters."},
    {"id": "Q08", "text": "EXP008 by Rohan used Transformer on TimeSeriesData-v3. MSE 0.018. Better than LSTM."},
    {"id": "Q09", "text": "NLP-Research project led by Ananya. Researchers: Ananya and Vikram. Domain: text classification."},
    {"id": "Q10", "text": "CV-Research project led by Priya. Researchers: Priya and Sneha. Domain: image recognition."},
    {"id": "Q11", "text": "TimeSeries-Research project led by Rohan. Domain: forecasting and anomaly detection."},
    {"id": "Q12", "text": "NLP-Corpus-v2 has 500k news documents. Split 80/10/10. Used in EXP001, EXP005, EXP006."},
]


# ============================================================
# SIMPLE RETRIEVER
# ============================================================

def embed(text, dim=48):
    text = text.lower()
    vec = [(text.count(chr(ord('a') + i % 26)) + i * 0.01) / (len(text) + 1)
           for i in range(dim)]
    norm = math.sqrt(sum(x**2 for x in vec)) + 1e-9
    return [x / norm for x in vec]

def sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x**2 for x in a)) + 1e-9) *
                  (math.sqrt(sum(x**2 for x in b)) + 1e-9))

def retrieve_chunks(query, docs, top_k=3):
    q = embed(query)
    scored = sorted(docs, key=lambda d: sim(q, embed(d["text"])), reverse=True)
    return scored[:top_k]


# ============================================================
# MOCK ANSWER GENERATOR
# ============================================================

def generate_answer(query, context_chunks):
    """
    Generates an answer from retrieved context.
    In production this calls Groq or another LLM.
    Here we extract key info from the best chunk.
    """
    if not context_chunks:
        return "No relevant information found in the experiment records."

    query_lower = query.lower()
    best = context_chunks[0]["text"]

    # build answer from most relevant chunk
    if any(w in query_lower for w in ["who", "researcher", "which researcher"]):
        match = re.search(r'by (\w+)', best)
        if match:
            return f"Based on experiment records: {match.group(1)} ran this experiment. Details: {best}"
    if any(w in query_lower for w in ["accuracy", "f1", "mse", "score", "result", "metric"]):
        return f"From experiment records: {best}"
    if any(w in query_lower for w in ["dataset", "data", "corpus"]):
        return f"Dataset information: {best}"
    if any(w in query_lower for w in ["model", "architecture", "bert", "lstm", "resnet"]):
        return f"Model details: {best}"
    if any(w in query_lower for w in ["project", "team", "lead"]):
        return f"Project information: {best}"

    return f"From ML experiment records: {best}"


# ============================================================
# STREAMING SIMULATION
# ============================================================

def stream_text(text, delay=0.03):
    """
    Simulates token-by-token streaming output.
    In production, Groq stream=True sends actual tokens.
    I added this to understand how streaming works.
    """
    words = text.split()
    output = []
    for i, word in enumerate(words):
        output.append(word)
        print(word, end=" ", flush=True)
        time.sleep(delay)
    print()   # newline after streaming
    return " ".join(output)


class StreamingRetriever:
    """
    Retrieves chunks and streams the answer word by word.
    Shows sources after the answer is complete.
    """
    def __init__(self, docs, stream_delay=0.02):
        self.docs = docs
        self.delay = stream_delay

    def query(self, question, top_k=3, stream=True):
        chunks = retrieve_chunks(question, self.docs, top_k)
        answer = generate_answer(question, chunks)

        result = {
            "question": question,
            "answer": answer,
            "sources": [c["id"] for c in chunks],
            "timestamp": datetime.now().isoformat()
        }

        if stream:
            print("\nAnswer: ", end="")
            stream_text(answer, delay=self.delay)
        else:
            print(f"\nAnswer: {answer}")

        print(f"Sources: {result['sources']}")
        return result


# ============================================================
# MULTI-TURN CONVERSATION
# ============================================================

class ConversationHistory:
    """
    Stores the conversation history for a session.
    Used to add context from previous turns to new queries.
    """
    def __init__(self, max_turns=10):
        self.turns = []
        self.max_turns = max_turns

    def add(self, question, answer):
        self.turns.append({"question": question, "answer": answer})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_context_string(self, n=2):
        """Returns last n turns as a string for query augmentation."""
        recent = self.turns[-n:]
        return " ".join(f"Previously: {t['question']}" for t in recent)

    def is_followup(self, query):
        """Detects if query is a follow-up to a previous question."""
        followup_words = ["it", "that", "this", "they", "them", "same", "also", "too",
                          "what about", "and", "how about", "more"]
        query_lower = query.lower()
        return any(query_lower.startswith(w) for w in followup_words)

    def summary(self):
        return {"total_turns": len(self.turns),
                "questions": [t["question"] for t in self.turns]}


class MultiTurnQA:
    """
    Multi-turn Q&A with context-aware retrieval.
    If a follow-up question is detected, previous context is
    appended to the query to improve retrieval accuracy.

    Example:
      Turn 1: "Which experiments did Ananya run?"
      Turn 2: "What accuracy did she get?"  <- follow-up, needs context
    """
    def __init__(self, docs, stream=True):
        self.retriever = StreamingRetriever(docs, stream_delay=0.02)
        self.history = ConversationHistory()
        self.stream = stream
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def ask(self, question):
        """Processes a single turn, adding history context if follow-up."""
        print(f"\n[Turn {len(self.history.turns) + 1}] Q: {question}")

        # augment query with history if follow-up
        if self.history.is_followup(question) and self.history.turns:
            context = self.history.get_context_string(n=2)
            augmented_query = context + " " + question
            print(f"[Context-aware query: {augmented_query[:80]}...]")
        else:
            augmented_query = question

        result = self.retriever.query(augmented_query, top_k=3, stream=self.stream)
        self.history.add(question, result["answer"])
        return result

    def run_session(self, questions):
        """Runs a full multi-turn session."""
        print("=" * 55)
        print(f"MULTI-TURN QA SESSION: {self.session_id}")
        print("=" * 55)
        results = []
        for q in questions:
            result = self.ask(q)
            results.append(result)
            print("-" * 40)
        print(f"\nSession complete. Total turns: {len(self.history.turns)}")
        return results

    def show_history(self):
        summary = self.history.summary()
        print(f"\nConversation History ({summary['total_turns']} turns):")
        for i, q in enumerate(summary["questions"]):
            print(f"  [{i+1}] {q}")


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("STREAMING QA APP DEMO")
    print("=" * 55)

    print("\n-- Part 1: Single streaming query --")
    retriever = StreamingRetriever(QA_KNOWLEDGE_BASE, stream_delay=0.01)
    retriever.query("Which experiments did Ananya run?", stream=True)

    print("\n-- Part 2: Multi-turn conversation --")
    multi_qa = MultiTurnQA(QA_KNOWLEDGE_BASE, stream=True)

    session_questions = [
        "Which researcher worked on NLP experiments?",
        "What accuracy did they achieve?",
        "Which datasets were used in those experiments?",
        "Who had the best F1 score?",
        "What model did they use?",
    ]

    multi_qa.run_session(session_questions)
    multi_qa.show_history()

    print("\n-- Part 3: Non-streaming comparison --")
    retriever_nostream = StreamingRetriever(QA_KNOWLEDGE_BASE, stream_delay=0)
    retriever_nostream.query("What was the best MSE in time series experiments?", stream=False)

    print("\n-- Streaming QA App demo complete --")


if __name__ == "__main__":
    run_demo()
