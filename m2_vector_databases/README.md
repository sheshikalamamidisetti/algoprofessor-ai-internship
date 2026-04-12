# Day 13: Vector Databases

**Author:** Sheshikala
**Date:** March 13, 2026

---

## Overview

For Day 13, I explored vector databases and how they enable semantic search by storing text as embeddings. Instead of relying on keyword matching, vector search compares the meaning of text by measuring similarity between numerical vectors.

To keep the workflow consistent with earlier days, I used machine learning experiment descriptions created previously and indexed them for semantic retrieval.

---

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the scripts in the following order:

```bash
python embedding_gen.py
python vector_db_setup.py
python hybrid_search.py
python interface_with_pinecone.py
```

> All scripts run locally and do not require Docker.

---

## Script Descriptions

### `embedding_gen.py`

Demonstrates how embeddings are generated and used for similarity search.

- Loads the `all-MiniLM-L6-v2` model from `sentence-transformers`
- Generates single and batch embeddings
- Computes cosine similarity between experiment descriptions
- Implements simple semantic search using NumPy
- Splits long documents into overlapping text chunks for better retrieval

### `vector_db_setup.py`

Demonstrates two different vector database approaches:

- **Chroma** — simple setup where text, embeddings, and metadata are stored together
- **FAISS** — manual embedding indexing with faster similarity search for larger datasets

Also includes a comparison of both systems.

### `hybrid_search.py`

Compares three retrieval approaches:

| Approach | Description |
|---|---|
| **BM25** | Keyword-based search, performs well for exact terms |
| **Semantic Search** | Embedding-based retrieval that captures meaning and paraphrases |
| **Hybrid (RRF)** | Combines both using Reciprocal Rank Fusion |

Several query types are tested to observe differences in retrieval quality.

### `interface_with_pinecone.py`

Demonstrates how a cloud-based vector database works using Pinecone.

- Runs in **demonstration mode** without an API key and prints operations that would occur in a real environment
- To use a live Pinecone database, create a free account and add your API key

---

## Key Concepts

### Embeddings

- Convert text into dense numerical vectors representing semantic meaning
- Texts with similar meaning produce vectors closer in vector space
- `all-MiniLM-L6-v2` generates embeddings with **384 dimensions**
- The same embedding model must be used during both indexing and querying

### Chroma vs FAISS

| | Chroma | FAISS |
|---|---|---|
| **Setup** | Simple, beginner-friendly | Manual embedding and metadata handling |
| **Metadata** | Supports filtering | Manual handling required |
| **Persistence** | SQLite-backed | Manual |
| **Performance** | Good for prototyping and RAG | Optimized for large-scale search |

### BM25 vs Semantic vs Hybrid Retrieval

- **BM25** works best when queries contain exact keywords
- **Semantic search** works better when queries are paraphrased or worded differently
- **Hybrid search** combines both and generally gives the most reliable results

### Text Chunking

- Long documents are split into smaller chunks before generating embeddings
- Prevents large documents from exceeding embedding limits and maintains search accuracy
- Typical config: **~200 words per chunk** with **~50 words of overlap**
- Overlap ensures important context is not lost at chunk boundaries

---

## Challenges Encountered

- Understanding how embeddings represent meaning in high-dimensional space required additional study
- FAISS requires manual vector normalization for cosine similarity, whereas Chroma handles this automatically
- The Reciprocal Rank Fusion formula required experimentation before its purpose became clear
- Setting up Pinecone in serverless mode required additional reading

---

## Connection to Other Days

| Day | Topic |
|---|---|
| **Day 12** | Experiment descriptions generated and stored in MongoDB — used as input here |
| **Day 14** | Builds a RAG pipeline using Chroma as a knowledge base |
| **Day 15** | Extends to Graph-RAG using Neo4j to incorporate relationships between documents |
