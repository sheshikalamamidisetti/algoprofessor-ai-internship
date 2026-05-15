# TalentTwin AI

TalentTwin AI is an AI-powered career intelligence and roadmap generation system built using modern Agentic AI frameworks. The project combines LLM reasoning, workflow orchestration, API engineering, and intelligent automation to help users explore career paths, identify skill gaps, and generate learning roadmaps.

## Project Overview

This project demonstrates practical implementation of:

* Agentic AI Systems
* Multi-step AI Workflows
* AI Reasoning Patterns
* API-based AI Applications
* LLM Integration
* FastAPI Backend Engineering
* LangGraph Orchestration

The system uses Groq LLMs with LangChain and LangGraph to create intelligent AI workflows capable of reasoning, reflection, and roadmap generation.

---

# Core Features

## AI Agents

### ReAct Agent

Implements the ReAct (Reasoning + Acting) pattern:

Thought → Action → Observation → Final Answer

The agent can:

* reason step-by-step
* use tools dynamically
* retrieve career information
* analyze skill gaps
* provide recommendations

### Reflexion Agent

Implements self-improving AI workflows:

* iterative reflection
* answer critique
* refinement cycles
* memory-based improvements

### TalentGraph Workflow

LangGraph-powered orchestration system:

* multi-step execution
* workflow state management
* structured AI pipelines
* roadmap generation logic

---

# API Features

The backend is built using FastAPI and includes:

* Health-check endpoints
* Async roadmap generation
* Job-based processing
* Swagger API documentation
* JSON request/response handling
* Interactive API testing

---

# API Endpoints

| Method | Endpoint                | Description                   |
| ------ | ----------------------- | ----------------------------- |
| GET    | `/`                     | API health check              |
| POST   | `/api/v1/roadmap`       | Generate career roadmap       |
| GET    | `/api/v1/jobs/{job_id}` | Get roadmap generation status |

---

# Technologies Used

## Languages

* Python
* SQL

## AI & ML

* LangChain
* LangGraph
* Groq API
* Pydantic

## Backend

* FastAPI
* Uvicorn
* Swagger UI

## Utilities

* DuckDuckGo Search
* JSON
* Async Processing

---

# Project Structure

```text id="5a0x8d"
TalentTwin/
│
├── agent.py
├── talent_graph.py
├── hitl_qa.py
├── main.py
├── server.py
├── requirements.txt
├── README.md
├── mcp_config.json
├── study_notes.md
└── agent_foundations/
```

---

# Setup Instructions

## 1. Install Dependencies

```bash id="8c1m2q"
pip install -r requirements.txt
```

## 2. Add Environment Variables

Create `.env` file:

```env id="6v7m3p"
GROQ_API_KEY=your_groq_api_key
```

## 3. Run FastAPI Server

```bash id="1q9m4x"
uvicorn main:app --reload
```

## 4. Open Swagger Docs

```text id="3m8x5v"
http://127.0.0.1:8000/docs
```

---

# Example Workflow

## Roadmap Generation

1. User submits career goal and skills
2. API validates request
3. TalentGraph workflow starts
4. AI agents process roadmap
5. Async job is created
6. User fetches generated roadmap using job ID

---

# Key Learnings

This project helped in understanding:

* Agentic AI architecture
* Workflow orchestration
* AI reasoning systems
* FastAPI backend development
* LLM integration
* Async API processing
* Multi-agent system design
* API testing using Swagger

---

# Future Improvements

* Frontend dashboard integration
* Database persistence
* Authentication system
* Cloud deployment
* Real-time streaming
* Multi-user support
* Advanced roadmap personalization
* Vector database integration

---

# Internship & Learning Context

Built as part of:

* AI R&D Internship
* IIT Indore Drishti CPS AI & DS Program
* Agentic AI and LLM Engineering Practice

---

