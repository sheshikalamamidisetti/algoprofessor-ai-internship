# Day 15: Graph RAG + Advanced RAG Techniques

**Author:** Sheshikala  
**Topic:** Knowledge graphs for multi-hop retrieval and advanced RAG techniques

---

## What I Built

Extended the Day 14 RAG pipeline with graph-based retrieval and advanced techniques.
Graph RAG stores relationships between researchers, experiments, datasets, and metrics
so we can answer multi-hop questions that vector search cannot handle.

---

## Folder Structure

```
day15/
  graph_rag.py          - ML knowledge graph with multi-hop queries
  advanced_rag.py       - HyDE, query expansion, reranking, multi-query, compression
  streaming_qa_app.py   - Streaming responses with multi-turn conversation
  day15_eval.ipynb      - Evaluation notebook: Recall@K, graph stats, technique comparison
  SETUP_GUIDE.sh        - Setup instructions
  requirements.txt      - Python dependencies
  README.md             - This file
```

---

## How to Run

Install dependencies:
```
pip install -r requirements.txt
```

Run individual files:
```
python graph_rag.py
python advanced_rag.py
python streaming_qa_app.py
```

Open evaluation notebook:
```
jupyter notebook day15_eval.ipynb
```

---

## Key Concepts Learned

| Concept | What I Learned |
|---|---|
| Graph RAG | Stores entity relationships for multi-hop queries |
| Multi-hop | Researcher -> Experiment -> Metric in one traversal |
| HyDE | Generate hypothetical answer, embed that instead of raw query |
| Query Expansion | Add synonyms to query for better recall |
| Reranking | Two-stage retrieval: fast bi-encoder then accurate cross-encoder |
| Multi-Query RAG | Generate sub-queries, retrieve each, merge with RRF |
| Contextual Compression | Keep only query-relevant sentences from each chunk |
| Streaming | Stream answer tokens as generated instead of waiting for full response |
| Multi-turn | Use conversation history to resolve follow-up questions |

---

## Neo4j Note

`graph_rag.py` includes an in-memory mock graph that runs without Neo4j.
To use real Neo4j: install Neo4j locally, start the service, then update
the URI and credentials in graph_rag.py.

---

## Connects To

- Day 13: Vector DB (ChromaDB, hybrid search)
- Day 14: RAG pipeline (chunking, embeddings, QA app)
