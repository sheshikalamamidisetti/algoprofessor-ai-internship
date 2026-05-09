# algoprofessor-ai-internship
AI R&D Internship — AlgoProfessor AI R&D Solutions (Feb–May 2026)
Intern: Sheshikala Mamidisetti | Track: Data Science & ML Agentic AI
IIT Indore Drishti CPS AI & Data Science Certification (Pursuing)

## ML Algorithm Coverage
Linear/Logistic Regression | Decision Tree | Random Forest | SVM
K-Means | PCA | LDA | SVD | XGBoost | LightGBM | PyTorch DL
ChromaDB | FAISS | BM25 | RAG | Graph RAG | Neo4j

## Progress Tracker
- [x] Day 01: Python Fundamentals — EDA Pipeline, OOP, NumPy, Pandas, Matplotlib
- [x] Day 02: Machine Learning — Random Forest, SVM, KMeans, Model Comparison
- [x] Day 03: Feature Engineering — XGBoost, LightGBM, PCA, LDA, SVD, PyTorch NN
- [x] Day 04: Deep Neural Networks — CNN, Transfer Learning, Training Pipeline
- [x] Day 05: NLP — Ollama, Llama3, Mistral, CoT, ReAct, DSPy Prompt Engineering
- [x] Day 06: LLM APIs — OpenAI, Claude, Function Calling, Pydantic, Memory, Time Series
- [ ] Day 07: RAG — FAISS, Embeddings, Retrieval Augmented Generation
- [ ] Day 08: Agent — ReAct Agent, LangChain, Tool Use
- [ ] Day 09: Multi-Agent — EDA Systems, CrewAI
- [x] Day 10: Milestone Project — DataOracle Capstone
- [x] Day 11: Databases — PostgreSQL, SQLAlchemy, pgvector, JSONB, Partitioning
- [x] Day 12: MongoDB + Redis — Document Store, Aggregation, Caching, Celery
- [ ] Day 13: Vector DBs — ChromaDB, FAISS, Pinecone, Hybrid Search BM25+Semantic
- [x] Day 14: RAG Pipeline — Chunking, Embeddings, Hybrid Retrieval, RAGAS Eval
- [x] Day 15: Graph RAG — Neo4j Knowledge Graph, HyDE, Reranking, Streaming QA
- [ ] Phase 2: LLM Engineering, Fine-tuning on Tabular/Time Series Data
- [ ] Phase 3: Agentic AI + Grand Capstone: DataSense AI

---

## Weekly Progress Update — Week 1 (Feb 22 – Feb 28)

### Completed Work

**Day 01 — Python Fundamentals (Iris Dataset)**
- Git and GitHub repository setup
- Python 3.10+ advanced syntax and OOP design patterns
- NumPy and Pandas EDA pipeline on Iris dataset
- Matplotlib and Seaborn dashboards — heatmap, boxplot, pairplot
- Automated EDA pipeline with Object-Oriented design

### Milestones Progress
- Phase 1 Foundations — In Progress
- GitHub repository initialized and EDA pipeline completed

### IIT Indore AI & Data Science Alignment
This week's internship work aligns with the following IIT Indore modules:
- Python for Data Science
- Exploratory Data Analysis
- Data Visualization Techniques
- Object-Oriented Programming

### Learning Outcomes
- Built automated EDA pipeline using OOP design
- Applied NumPy and Pandas for data analysis
- Created professional visualizations using Matplotlib and Seaborn
- Set up Git and GitHub for version-controlled project management

---

## Weekly Progress Update — Week 2 (Mar 1 – Mar 7)

### Completed Work

**Day 02 — Machine Learning Models (Heart Disease and Breast Cancer Dataset)**
- Linear Regression and Logistic Regression on Breast Cancer dataset
- Decision Tree, Random Forest — Accuracy 0.80, ROC-AUC 0.91
- SVM Classification — Accuracy 0.82, ROC-AUC 0.883
- KMeans Clustering — Silhouette Score 0.167, Optimal K=2
- Hyperparameter Tuning on Decision Tree
- Model Comparison — SVM best overall accuracy

**Day 03 — Feature Engineering and Advanced ML (Wine Quality Dataset)**
- XGBoost Accuracy 0.825, ROC-AUC 0.881 vs LightGBM Accuracy 0.790
- Feature Engineering — 4 new features created, +1.56% accuracy improvement
- Dimensionality Reduction — PCA, LDA, SVD — Best: LDA Accuracy 0.7125
- Production-grade Scikit-learn Pipeline — Best: RF Accuracy 0.803, ROC-AUC 0.902
- PyTorch Neural Network from Scratch — Accuracy 0.759
- Auto Report Generation

### Milestones Progress
- M1: Web Intelligence Synthesiser — Completed
- All supervised and unsupervised ML models completed
- Production-grade pipelines with XGBoost and LightGBM completed

### IIT Indore AI & Data Science Alignment
This week's internship work aligns with the following IIT Indore modules:
- Supervised Learning — Classification and Regression
- Unsupervised Learning — Clustering
- Model Evaluation and Comparison
- Ensemble Methods — XGBoost and LightGBM
- Feature Engineering and Selection
- Dimensionality Reduction — PCA, LDA, SVD
- Deep Learning Foundations with PyTorch
- Production ML Pipeline Design

### Learning Outcomes
- Implemented 7 ML models with full evaluation metrics
- Applied ensemble methods XGBoost and LightGBM
- Built engineered features improving accuracy by +1.56%
- Reduced dimensions using PCA, LDA and SVD techniques
- Built production-grade Scikit-learn pipelines
- Implemented Neural Network from scratch using PyTorch
- Practiced professional GitHub commit workflow

---

## Weekly Progress Update — Week 3 (Mar 8 – Mar 14)

### Completed Work

**Day 04 — Deep Neural Networks (Image Classification)**
- CNN Classifier with convolutional and pooling layers
- Neural Network built from scratch using PyTorch
- Transfer Learning with pretrained model fine-tuning
- Training pipeline with validation and early stopping
- Statistical theory grounding with Linear Algebra foundations

**Day 11 — PostgreSQL Databases (ML Experiment Tracker)**
- Schema design for researchers, projects, datasets, experiments, metrics
- Complex SQL queries — JOINs, CTEs, Window Functions, aggregations
- JSONB storage for flexible ML hyperparameters with GIN indexing
- Table partitioning by date range and hash partitioning
- Advanced indexing strategies — B-tree, GIN, partial indexes
- SQLAlchemy ORM with async queries and pgvector extension

**Day 12 — MongoDB + Redis (ML Experiment Logs)**
- MongoDB CRUD operations on experiment collections
- Aggregation pipelines — $group, $facet, $lookup for F1 score analysis
- Redis cache-aside pattern — 200x speedup on repeated queries
- TTL-based cache expiry for experiment metadata
- Celery task queue for async experiment processing
- Cache benchmark comparing MongoDB vs Redis latency

**Day 13 — Vector Databases (ML Knowledge Base)**
- ChromaDB setup with cosine similarity and persistent storage
- FAISS IndexFlatL2 for exact search and IndexIVFFlat for approximate search
- Pinecone interface with demo mode fallback
- Hybrid search combining BM25 keyword search with semantic vector search
- Reciprocal Rank Fusion merging BM25 and semantic rankings
- Embedding generation with sentence-transformers all-MiniLM-L6-v2

**Day 14 — RAG Pipeline (ML Experiment Tracker Knowledge Base)**
- 5 chunking strategies — fixed size, sentence, paragraph, recursive, semantic
- Document ingestion pipeline — load, clean, chunk, embed, store
- Basic RAG loop — index, retrieve, generate with retrieval recall evaluation
- Hybrid RAG pipeline with BM25 and semantic search with RRF fusion
- RAGAS-style evaluation — faithfulness, answer relevancy, context precision, recall
- Groq LLM integration with mock fallback for generation step

**Day 15 — Graph RAG and Advanced RAG (ML Knowledge Graph)**
- ML knowledge graph with 5 node types — Researcher, Project, Dataset, Experiment, Metric
- Multi-hop graph queries — Researcher to Experiment to Metric traversal
- 5 advanced RAG techniques — HyDE, query expansion, cross-encoder reranking,
  multi-query retrieval, contextual compression
- Streaming QA app with token-by-token output simulation
- Multi-turn conversation with context-aware retrieval
- Recall@1 and Recall@3 evaluation on 7 test cases

### Milestones Progress
- M2: Enterprise Knowledge Navigator — Completed
- Deep neural networks, PostgreSQL, MongoDB, Redis, Vector DBs, RAG, Graph RAG all implemented
- Full Week 3 knowledge pipeline from deep learning to Graph RAG completed

### IIT Indore AI & Data Science Alignment
This week's internship work aligns with the following IIT Indore modules:
- Deep Learning and Neural Networks
- Database Systems — Relational and NoSQL
- Information Retrieval and Search
- Natural Language Processing — Embeddings and Semantic Search
- Knowledge Representation — Graph Databases
- Applied Machine Learning — Retrieval Augmented Generation
- Evaluation Metrics for AI Systems

### Learning Outcomes
- Built CNN and applied transfer learning for image classification
- Designed and queried PostgreSQL schemas with advanced indexing
- Implemented MongoDB aggregation pipelines for experiment analytics
- Built Redis caching layer achieving 200x query speedup
- Set up ChromaDB and FAISS vector stores with hybrid search
- Built complete RAG pipeline from chunking to RAGAS evaluation
- Implemented Graph RAG with multi-hop Neo4j-style knowledge graph
- Applied 5 advanced RAG techniques including HyDE and cross-encoder reranking
- Built streaming QA app with multi-turn conversational memory

---

## Weekly Progress Update — Week 4 (Mar 15 – Mar 24)

### Completed Work

**Day 05 NLP — Local LLMs and Prompt Engineering (Titanic Dataset)**
- Ollama client setup for running Llama3 and Mistral locally without API costs
- Llama3 pipeline for Titanic dataset analysis and NLP report generation
- Mistral pipeline for concise outputs and side-by-side model comparison
- Chain of Thought prompting for step-by-step data analysis reasoning
- ReAct agent with 5 data analysis tools for multi-step reasoning
- DSPy-style modular prompt pipeline with Predict and ChainOfThought modules
- Few-shot vs zero-shot comparison demonstrating prompt quality improvement

**Day 06 DataAssist Analytics Agent — LLM APIs and Structured Outputs (Titanic + Time Series)**
- OpenAI API client for data analytics with mock fallback
- Claude API client for data analytics with mock fallback
- Function calling with structured JSON tool definitions and dispatch
- Pydantic schemas for validated structured report outputs
- Automated data analysis report generator using LLM and dataset statistics
- Conversational memory manager for multi-turn analytics sessions
- Time series analyzer — trend detection, anomaly detection, 7-day forecasting
- Full DataAssist Analytics Agent combining all components

### Milestones Progress
- M3: DataAssist Analytics Agent — Completed
- Local LLM integration with Ollama completed for Llama3 and Mistral
- Prompt engineering techniques implemented — CoT, ReAct, DSPy
- OpenAI and Claude API integration with function calling completed
- Time series analysis with LLM narration completed

### IIT Indore AI & Data Science Alignment
This week's internship work aligns with the following IIT Indore modules:
- Large Language Models and Prompt Engineering
- Chain of Thought and ReAct Reasoning Patterns
- Function Calling and Structured Outputs
- Conversational AI and Memory Systems
- Time Series Analysis and Forecasting
- Applied AI for Data Analytics

### Learning Outcomes
- Ran Llama3 and Mistral locally via Ollama without API costs
- Applied Chain of Thought prompting to improve analytical reasoning accuracy
- Built ReAct agent with real data analysis tools for multi-step queries
- Implemented DSPy-style modular prompt engineering with few-shot examples
- Integrated OpenAI and Claude APIs for structured data analytics
- Built function calling system with JSON tool schemas
- Created Pydantic models for validated structured LLM outputs
- Built conversational memory for context-aware multi-turn sessions
- Implemented time series trend detection and LLM-narrated forecasting
 
----
 
 ## Weekly Progress Update — Week 5 (Mar 25 – Mar 31)

### Completed Work

#### Milestone 4 — InsightScribe (Days 26–28)

* Built end-to-end audio AI pipeline:

  * gTTS audio generation
  * Whisper base model transcription
* Implemented speaker diarisation (2 speakers with timestamps)
* KPI extraction using regex (revenue, percentages, user counts)
* 3-way LLM comparison:

  * GPT-4o vs Claude 3.5 Sonnet vs Llama3 (Groq)
* Structured output using Pydantic v2
* Automated PDF report generation using ReportLab
* FastAPI REST API (5 endpoints tested)
* Docker containerization
* Gradio UI:

  * Upload audio → process → view results

#### Milestone 5 — CodeXcelerate (Days 29–30)

* pytest suite:

  * 16 tests
  * 87% coverage
* LLM evaluation system:

  * Faithfulness score
  * Relevancy score
  * Hallucination detection
* NeMo Guardrails:

  * Topic filtering
  * PII protection
  * Hallucination blocking
* FastAPI APIs for evaluation and guardrails
* Benchmark reports:

  * CSV + JSON + HTML outputs
* mkdocs documentation (Material theme)

### Milestones Progress

* M4: InsightScribe — Completed
* M5: CodeXcelerate — Completed
* Phase 1 — Completed

### IIT Indore AI & Data Science Alignment

* Audio Processing — Whisper
* LLM APIs — OpenAI, Claude, Groq
* Structured Outputs — Pydantic v2
* API Development — FastAPI
* Testing — pytest
* AI Safety — NeMo Guardrails
* Deployment — Docker and Gradio

### Learning Outcomes

* Built audio-to-insight pipeline using Whisper and LLMs
* Compared multiple LLM outputs for better decision making
* Implemented structured validated reports using Pydantic
* Developed REST APIs with Docker deployment
* Designed LLM evaluation metrics and safety guardrails
* Created interactive UI for real-time AI processing

---

## Weekly Progress Update — Week 6 (Apr 1 – Apr 7)

### Completed Work

#### Milestone 6 — DataOracle (Days 31–35)

* Built multi-LLM benchmarking system:

  * GPT-4o, Claude, Gemini, Llama
* Unified model interface (model_registry)
* Benchmark runner:

  * Tasks: statistical inference, ML code, EDA
  * Outputs: CSV and JSON benchmark reports
* Tree-of-Thought reasoning pipeline (DSPy):

  * Explores multiple reasoning paths
  * Selects best statistical solution
* Pydantic v2 validation layer:

  * Validates all LLM outputs
  * Prevents invalid or hallucinated values
* Automated ML insights report generator:

  * Markdown, JSON and PDF outputs
* Gradio web app (4 tabs):

  * Benchmarking
  * ToT reasoning
  * Report generation
  * System overview
* pytest testing for schemas and pipeline

### Milestones Progress

* M6: DataOracle — Completed
* Phase 2 — In Progress

### IIT Indore AI & Data Science Alignment

* LLM Engineering — Multi-model benchmarking
* Inferential Statistics — Hypothesis testing tasks
* Advanced AI Reasoning — Tree-of-Thought (DSPy)
* Data Engineering — Pydantic validation
* ML Evaluation — Metrics and scoring
* Data Communication — Automated reports
* Deployment — Gradio UI

### Learning Outcomes

* Learned that different LLMs perform better on different tasks
* Implemented Tree-of-Thought for improved reasoning accuracy
* Built production-level validation using Pydantic
* Designed benchmarking system for real-world data science tasks
* Automated end-to-end ML reporting pipeline
* Built full-stack AI system with UI, backend and evaluation


## Weekly Progress Update — Week 7 (Apr 8 – Apr 14)

### Completed Work

**Milestone 7 — TimeSeriesHunter (Days 36–40)**

LoRA/QLoRA fine-tuning setup for Llama 3.1 8B:

- BitsAndBytes 4-bit NF4 quantisation — reduces VRAM from 80GB to 8GB
- PEFT LoRA config — rank=16, alpha=32, target attention layers only
- Trainable parameters: 0.08% of total (6M out of 8B)

TRL SFTTrainer on tabular and time series datasets:

- Instruction pairs covering ARIMA selection, trend analysis, seasonality detection
- Llama 3.1 chat format with system, user, and assistant tokens
- Dry-run validation pipeline (no GPU needed)

DPO preference tuning for data analyst behaviour:

- Chosen vs rejected response pairs for 4 DS scenarios
- Trains model to prefer precise statistical answers over vague ones
- Beta=0.1 temperature for stable preference learning

GPTQ/AWQ quantisation and vLLM inference:

- AWQ activation-aware quantisation — 4GB compressed from 16GB
- vLLM continuous batching — 10x throughput vs HuggingFace generate
- GPTQ vs AWQ comparison with cost analysis

W&B experiment tracking:

- Tracks train/loss, learning rate, eval scores per task
- Model artifact logging for reproducibility
- Milestone 7 evaluation on 5 time series DS tasks — 86% average score

### Milestones Progress

- [ ] M7: TimeSeriesHunter — QLoRA Llama 3.1

### IIT Indore AI & Data Science Alignment

This week's internship work aligns with the following IIT Indore modules:

- Time Series Analysis — ARIMA, stationarity, ACF/PACF tasks
- Deep Learning Architecture — LoRA rank, adapter weights, attention layers
- Supervised Learning — SFT instruction pairs
- Reinforcement Learning from Human Feedback — DPO preference optimisation
- Model Deployment and Optimisation — AWQ quantisation, vLLM serving
- ML Experiment Management — W&B experiment tracking

### Learning Outcomes

- Fine-tuned Llama 3.1 8B on a single GPU using QLoRA 4-bit compression
- Applied SFT to teach correct data science terminology and methods
- Used DPO to shape model behaviour toward expert data analyst responses
- Quantised fine-tuned model to AWQ 4-bit for efficient deployment
- Served quantised model with vLLM achieving 10x throughput improvement
- Tracked all experiments with W&B for full reproducibility

---

## Weekly Progress Update — Week 8 (Apr 15 – Apr 21)

### Completed Work

**Milestone 8 — InsightBot (Days 41–45)**

RAG vs fine-tuning comparison for DS knowledge bases:

- Side-by-side comparison of plain LLM vs RAG on DS queries
- Decision guide — when to use RAG, fine-tuning, or both
- FAISS keyword fallback for environments without GPU

GPT-4V and LLaVA multimodal chart and dashboard analysis:

- Generated time series and model comparison charts using Matplotlib/Seaborn
- GPT-4V vision analysis extracting trends, anomalies, and recommendations
- LLaVA local model alternative (no API cost)
- Extends existing Power BI and Matplotlib/Seaborn skills

pgvector embeddings for statistical reports:

- Stored 4 DS report types in PostgreSQL with vector embeddings
- Semantic search returning similarity scores per query
- Comparison of pgvector vs FAISS for production use cases

MT-Bench multi-turn evaluation and HELM metrics:

- 3 multi-turn DS tasks testing reasoning across conversation turns
- HELM metrics — accuracy, calibration, robustness, fairness, efficiency, toxicity
- LangSmith tracing setup for agent call monitoring
- Arize Phoenix local dashboard for embedding drift detection

InsightBot 5-agent CrewAI pipeline:

- DataRetriever — semantic search over DS knowledge base via pgvector
- ChartAnalyst — GPT-4V reads and interprets charts and dashboards
- StatReasoner — Tree-of-Thought statistical reasoning
- ReportWriter — synthesises all findings into structured report
- QualityChecker — MT-Bench style quality scoring and feedback
- Average quality score 8.5/10 across 5 DS evaluation queries

### Milestones Progress

- [ ] M8: InsightBot — Multi-Agent Data Analyst

### IIT Indore AI & Data Science Alignment

This week's internship work aligns with the following IIT Indore modules:

- Information Retrieval — RAG vs fine-tuning decision framework
- Computer Vision and Multimodal AI — GPT-4V chart analysis
- Inferential Statistics — Tree-of-Thought reasoning in StatReasoner
- Data Science Communication — structured report synthesis
- LLM Evaluation — MT-Bench and HELM metrics
- Multi-Agent Systems — CrewAI 5-agent orchestration
- MLOps and Observability — LangSmith tracing, Arize Phoenix monitoring

### Learning Outcomes

- Built RAG pipeline extended from day7_rag with pgvector for persistent storage
- Analysed generated charts using GPT-4V multimodal vision capabilities
- Evaluated LLM quality using MT-Bench multi-turn tasks and HELM dimensions
- Designed 5-agent CrewAI system with specialised roles for each DS workflow step
- Integrated LangSmith and Arize Phoenix for production-grade observability
- Achieved 8.5/10 quality score across all InsightBot evaluation queries

---


## Grand Capstone
DataSense AI — Autonomous Intelligent Data Analysis and Insights Platform
4 Agents | 6 MCP Servers | 40+ Tools | SQL + Power BI + ML Fusion
