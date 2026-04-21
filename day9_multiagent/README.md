# day9_multiagent/ — InsightBot: Multi-Agent Data Analyst
### Milestone 8 | Week 8 | Days 41–45 | April 15–21
### Sheshikala | IIT Indore AI & Data Science Programme

---

## About

InsightBot is a five-agent CrewAI system built for data science workflows.
Instead of asking one LLM to do everything, each agent handles one specialised
job — retrieval, visual analysis, statistical reasoning, report writing, and
quality checking. The agents run sequentially, each building on the previous
agent's output, producing a final report that covers every dimension of a DS
analyst's response.

This milestone integrates all Phase 2 components — RAG from `day7_rag/`,
Tree-of-Thought reasoning from `day10_project/`, pgvector from `day11_databases/`,
and GPT-4V multimodal analysis — into one working system.

---

## Folder Structure

```
day9_multiagent/
├── eval_observability.py          Day 44 — MT-Bench + HELM + LangSmith + Arize
├── insightbot.py                  Day 45 — full 5-agent pipeline (M8 deliverable)
├── tests/
│   └── test_week8.py
├── notebooks/
│   └── insightbot_demo.ipynb
├── outputs/
│   ├── InsightBot_M8_*.md         main milestone report
│   └── InsightBot_M8_*.json
└── requirements.txt

day7_rag/  (extended — also part of Week 8)
├── rag_vs_finetuning.py           Day 41
├── multimodal_chart_analysis.py   Day 42
└── pgvector_embeddings.py         Day 43
```

---

## The Five Agents

| Agent | Job |
|-------|-----|
| DataRetriever | Semantic search over DS knowledge base via pgvector |
| ChartAnalyst | GPT-4V reads and interprets charts and dashboards |
| StatReasoner | Tree-of-Thought statistical reasoning |
| ReportWriter | Synthesises all findings into a structured report |
| QualityChecker | MT-Bench style quality scoring and feedback |

---

## How to Run

```bash
pip install -r requirements.txt

# Demo — no API key needed
python insightbot.py --milestone

# With API key
python insightbot.py --query "analyse churn data"

# Tests
pytest tests/ -v
```

---

## IIT Indore Curriculum Alignment

| Component | Module |
|-----------|--------|
| pgvector RAG retrieval | Information Retrieval |
| GPT-4V multimodal analysis | Computer Vision, Multimodal AI |
| Tree-of-Thought reasoning | Inferential Statistics |
| Structured report synthesis | Data Science Communication |
| MT-Bench / HELM evaluation | LLM Evaluation |
| CrewAI multi-agent orchestration | Multi-Agent Systems |
| LangSmith + Arize Phoenix | MLOps and Observability |
