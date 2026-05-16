# DataSquad AI

DataSquad AI is a multi-agent AI-powered analytics and machine learning system built using Python and modern data science libraries. The project combines automated EDA pipelines, PCA dimensionality reduction, clustering orchestration, and AI-inspired workflow automation to simulate a collaborative data science team.

## Project Overview

This project demonstrates practical implementation of:

* Multi-Agent AI Systems
* Automated Data Analytics
* Exploratory Data Analysis (EDA)
* Machine Learning Pipelines
* PCA Dimensionality Reduction
* KMeans Clustering Orchestration
* AI-driven Report Generation
* Workflow-based Analytics Automation

The system uses specialized analytics agents that collaborate together to process datasets, generate insights, perform clustering, and produce reports automatically.

---

# Core Features

## AI Analytics Agents

### DataPlanner

Handles dataset understanding and analytics planning:

* loads datasets
* identifies feature types
* detects clustering-ready columns
* defines analysis strategy

### StatAnalyst

Runs the automated EDA pipeline:

* generates statistical summaries
* creates visualizations
* detects outliers
* computes correlations
* performs profiling

### MLEngineer

Handles machine learning workflows:

* applies PCA dimensionality reduction
* runs KMeans clustering
* tests multiple K values
* selects optimal cluster count
* generates clustering diagnostics

### ReportWriter

Compiles final analytics outputs:

* generates HTML reports
* exports JSON summaries
* organizes visual outputs
* creates analytics dashboard files

---

# Machine Learning Features

## PCA (Principal Component Analysis)

Used for:

* dimensionality reduction
* variance retention
* feature compression
* visualization optimization

## KMeans Clustering

Used for:

* customer segmentation
* pattern discovery
* unsupervised learning
* cluster analysis

The system automatically:

* tests multiple K values
* computes silhouette scores
* selects optimal cluster count
* visualizes cluster distributions

---

# Automated EDA Pipeline

The project automatically generates:

* distribution plots
* correlation heatmaps
* boxplots
* categorical analysis charts
* outlier detection
* descriptive statistics

---

# Technologies Used

## Languages

* Python

## Data Science & ML

* Pandas
* NumPy
* Scikit-learn
* Seaborn
* Matplotlib

## Utilities

* AsyncIO
* JSON
* Pathlib

---

# Project Structure

```text
day15_datasquad/
│
├── main.py
├── data_planner.py
├── stat_analyst.py
├── ml_engineer.py
├── report_writer.py
├── clustering_tool.py
├── pca_tool.py
├── eda_pipeline.py
├── logger.py
├── requirements.txt
├── sample_customers.csv
├── README.md
│
├── outputs/
│   ├── datasquad_report.html
│   └── datasquad_summary.json
│
├── screenshots/
│   ├── output1.png
│   └── output2.png
```

---

# Setup Instructions

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run Project

```bash
python main.py
```

---

# Example Workflow

## Analytics Pipeline

Dataset Input
↓
Data Planning
↓
EDA Analysis
↓
PCA Reduction
↓
KMeans Clustering
↓
HTML Report Generation

---

# Outputs

The system generates:

* HTML analytics report
* JSON summary
* EDA charts
* PCA diagnostics
* clustering visualizations
* statistical summaries

Generated outputs are stored inside:

```text
outputs/
```

---

# Key Learnings

This project helped in understanding:

* Multi-agent system design
* Automated analytics workflows
* PCA and clustering pipelines
* Machine learning orchestration
* Python data science architecture
* AI-inspired workflow automation
* Statistical analysis pipelines
* Report generation systems

---

# Future Improvements

* LangGraph integration
* CrewAI integration
* SQL/BI server connectivity
* Streamlit dashboard
* Cloud deployment
* Real-time analytics
* LLM-powered insights
* Docker support

---

# Internship & Learning Context

Built as part of:

* AI & Data Science Internship
* Agentic AI Learning Program
* Multi-Agent Systems Practice
* Machine Learning Workflow Engineering

