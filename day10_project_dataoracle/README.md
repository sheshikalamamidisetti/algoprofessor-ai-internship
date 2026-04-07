# day10_project — DataOracle: LLM Insights Predictor
### Milestone 6 | Phase 2 | Week 6 | Days 31–35 | April 1–7
### Sheshikala | IIT Indore AI & Data Science Programme

---

## What Is This Milestone?

**DataOracle** is a system that benchmarks multiple LLMs (GPT-4o, Claude, Gemini, Llama)
on real data science tasks, reasons through statistical problems using Tree-of-Thought,
validates all outputs using Pydantic AI, and generates an automated ML insights report.

The core question this milestone answers:

> *"Which LLM is best for data science work, and how do we make it produce
> structured, validated, reliable outputs for a DS analyst?"*

---

## Why We Built This

By Day 30 you had built individual tools — RAG, agents, multi-agent systems.
Milestone 6 integrates them into one cohesive **intelligence layer** that:

- Does not rely on one LLM — it **compares all four** objectively
- Does not trust raw LLM output — it **validates everything with Pydantic**
- Does not just answer questions — it **reasons step-by-step** (Tree-of-Thought)
- Does not produce prose — it **generates structured reports** automatically

This directly aligns with the IIT Indore AI & DS curriculum because every component
maps to a core module — inferential statistics (ToT tasks), ML model evaluation
(Pydantic schemas), LLM engineering (benchmarking), and data communication (automated report).

---

## Folder Structure

```
day10_project/
│
├── model_registry.py          Day 31 — unified GPT-4o/Claude/Gemini/Llama wrapper
├── benchmark_runner.py        Day 32 — DS benchmark runner, saves CSV + JSON
├── tot_dspy_pipeline.py       Day 33 — Tree-of-Thought + DSPy reasoning
├── pydantic_schemas.py        Day 34 — Pydantic v2 validated structured outputs
├── ml_report_generator.py     Day 35 — automated report (Milestone deliverable)
├── app.py                     Day 35 — Gradio UI integrating all 5 components
│
├── notebooks/
│   └── dataoracle_demo.ipynb  end-to-end notebook (run all 5 days in one place)
│
├── tests/
│   └── test_dataoracle.py     pytest unit tests — schemas, scorer, report
│
├── outputs/                   generated reports, CSVs, charts saved here
├── requirements.txt
└── .env.example               API key template
```

---

## What Each File Does and Why

### `model_registry.py` — Day 31

**What:** A single unified wrapper so that `model.chat("your prompt")` works
identically for GPT-4o, Claude, Gemini, and Llama.

**Why:** Without this, benchmark code would have four different API call styles.
With this, you swap models by changing one string. This is the engineering
discipline pattern — program to an interface, not an implementation.

**Output:** `ModelResponse` dataclass with `.response`, `.tokens`, `.latency_ms`

---

### `benchmark_runner.py` — Day 32

**What:** Runs three categories of DS tasks across all four LLMs and scores each response
against a rubric.

**Task categories:**
- `stat_inference` — hypothesis testing, confidence intervals, ANOVA interpretation
- `ml_code` — sklearn pipelines, overfitting diagnosis with fixes
- `eda` — full pandas EDA plan for a given dataset

**Why:** You need objective evidence of which LLM performs best on *your specific use case*
(data science), not just general benchmarks. This gives you that evidence with real scores.

**Output:** `outputs/benchmark_<timestamp>.csv` and `.json`

---

### `tot_dspy_pipeline.py` — Day 33

**What:** Tree-of-Thought reasoning using DSPy. For any statistical problem,
it explores three reasoning branches (parametric / non-parametric / Bayesian),
scores each one, and picks the most statistically sound answer.

**Why:** A data scientist doesn't just answer immediately — they consider multiple
approaches, justify their choice, then commit. This is exactly how a senior statistician
reasons. Direct prompting gives one answer. ToT gives you the *best* answer with justification.

**Why DSPy:** DSPy compiles the reasoning chain so you can optimise and reuse it.
Instead of brittle prompt strings, you have typed signatures with input/output fields.

**Output:** A traced reasoning result with `branches_explored`, `best_branch`, `justification`, `final_answer`

---

### `pydantic_schemas.py` — Day 34

**What:** Pydantic v2 data models that validate every LLM output before it is used.

**Schemas:**
- `HypothesisTestResult` — validates p-value is 0–1, checks all fields are present
- `ModelMetrics` — accuracy, F1, RMSE all in valid ranges
- `MLModelEval` — full model evaluation with overfitting flag
- `LLMBenchmarkEntry` — one scored result per LLM per task
- `DataOracleReport` — the master report combining everything

**Why:** LLMs hallucinate. An LLM might return `p_value=1.3` or skip the conclusion field.
Pydantic catches this immediately and raises a `ValidationError` instead of silently
passing bad data downstream. Every field that reaches the report has been validated.

**Output:** Importable schema classes used by `benchmark_runner.py`, `reporter.py`, and the tests

---

### `ml_report_generator.py` — Day 35

**What:** The Milestone 6 deliverable. Takes benchmark results, calls an LLM to
generate insights, validates everything through Pydantic schemas, and saves
a structured report as Markdown + JSON + PDF.

**Why:** This is the end product of DataOracle — a report that a DS team could actually
use. It answers: which LLM performed best, what were the key findings, what should
the team do next. It is automated — you run one command and get a professional report.

**Output:**
```
outputs/DataOracle_Report_<timestamp>.md
outputs/DataOracle_Report_<timestamp>.json
outputs/DataOracle_Report_<timestamp>.pdf   (if fpdf2 installed)
```

---

### `app.py` — Day 35 (capstone integration)

**What:** A Gradio web application with four tabs that integrates all five components.

```
Tab 1 — LLM Benchmark     run benchmark_runner on any model combination
Tab 2 — ToT Reasoning     run tot_dspy_pipeline on any pre-built problem
Tab 3 — Generate Report   run ml_report_generator, preview output in browser
Tab 4 — About             architecture diagram and student info
```

**Why:** Proves the system works end-to-end through a UI, not just individual scripts.
A reviewer can open `localhost:7860` and interact with DataOracle without touching the terminal.

---

## Setup

```bash
cd day10_project
pip install -r requirements.txt

cp .env.example .env
# open .env and add your API keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...
```

For local Llama (optional):
```bash
# Install Ollama from https://ollama.com
ollama pull llama3
ollama serve
```

---

## How to Run

```bash
# Validate all Pydantic schemas (no API key needed)
python pydantic_schemas.py

# Run benchmarks
python benchmark_runner.py --models gpt4o claude --tasks stat_inference ml_code

# Run Tree-of-Thought reasoning
python tot_dspy_pipeline.py --problem ttest

# Generate milestone report
python ml_report_generator.py
python ml_report_generator.py --results outputs/benchmark_<timestamp>.json

# Run all tests
pytest tests/ -v

# Launch Gradio app
python app.py
# open http://localhost:7860
```

---

## Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/llm-engineering-60days/day10_project')

!pip install -r requirements.txt -q
!python benchmark_runner.py --models gpt4o claude --tasks stat_inference
```

Open the notebook directly:
`notebooks/dataoracle_demo.ipynb` → File → Open in Colab

---

## IIT Indore Curriculum Alignment

| DataOracle component | Curriculum module |
|---------------------|------------------|
| Hypothesis testing tasks in benchmark | Inferential Statistics |
| Model accuracy / F1 / RMSE schemas | ML Model Assessment |
| LLM API calls (OpenAI, Anthropic, Google) | LLM Engineering |
| Pydantic validated structured outputs | Data Engineering |
| Tree-of-Thought multi-step reasoning | Advanced AI Reasoning |
| Automated markdown + PDF report | Data Science Communication |
| Gradio UI deployment | Applied ML Deployment |

---

## Key Learning from This Milestone

**Benchmarking** teaches you that no single LLM is best at everything.
Claude excels at multi-step statistical reasoning. GPT-4o at code generation.
Gemini at speed. Llama is free and local. A real DS team picks the right tool for each job.

**Tree-of-Thought** teaches you that how you ask is as important as what you ask.
Exploring three reasoning branches before committing produces better statistical answers
than direct prompting — measurably, by rubric score.

**Pydantic validation** teaches you that LLMs are unreliable data sources.
Treating LLM output as untrusted external input — validating every field before use —
is the difference between a toy project and production-ready code.

**Automated reporting** teaches you that the end product of data science is communication.
A pipeline that runs end-to-end and produces a professional report that a non-technical
stakeholder can read is worth more than a notebook full of code cells.
