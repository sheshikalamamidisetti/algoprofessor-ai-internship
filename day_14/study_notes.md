# Days 61–65 Study Notes & Architecture

## D61 — ReAct Agent + SQL MCP
**Pattern**: Thought → Action → Observation loop
- File: `agent_foundations/react_agent/agent.py`
- MCP: `mcp_servers/sql_connector/server.py`
- Key concept: Tools give the agent grounded, real-world data access
- Run: `python agent_foundations/react_agent/agent.py`

## D62 — Reflexion Agent + REST Analytics MCP
**Pattern**: Generate → Evaluate → Self-critique → Retry
- File: `agent_foundations/reflexion_agent/agent.py`
- MCP: `mcp_servers/rest_analytics/server.py`
- Key concept: LLM as its own critic; iterative quality improvement
- Run: `python agent_foundations/reflexion_agent/agent.py`

## D63 — Plan-Execute Agent + Power BI MCP
**Pattern**: Planner → Executor → (optional) Replanner
- File: `agent_foundations/plan_execute/agent.py`
- MCP: `mcp_servers/powerbi_server/server.py`
- Key concept: Separation of planning vs execution; Replanner for dynamic adjustment
- Run: `python agent_foundations/plan_execute/agent.py`

## D64 — LangGraph State Machines + Stat Model Store
**Pattern**: Explicit state graph with typed state dict
- File: `langgraph/state_machines/talent_graph.py`
- MCP: `mcp_servers/stat_model_store/server.py`
- Key concept: Deterministic node transitions; conditional edges for loops
- Run: `python langgraph/state_machines/talent_graph.py`

## D65 — Human-in-the-Loop QA + TalentTwin Integration
**Pattern**: interrupt() pauses graph → human resumes with Command()
- File: `langgraph/human_in_loop/hitl_qa.py`
- Key concept: LangGraph checkpointing enables pause/resume across process restarts
- Run: `python langgraph/human_in_loop/hitl_qa.py`

---

## Architecture Overview

```
User Input
    │
    ▼
FastAPI (port 8000)
    │
    ├── POST /api/v1/roadmap ──► LangGraph Pipeline (async job)
    │       │
    │       ├── intake_node
    │       ├── profile_build_node
    │       ├── skill_analysis_node ──► SQL MCP (skill taxonomy)
    │       ├── market_research_node ──► REST Analytics MCP
    │       ├── gap_analysis_node
    │       ├── roadmap_generation_node ──► Stat Model Store MCP
    │       └── qa_review_node ──► [HITL interrupt if enabled]
    │
    ├── POST /api/v1/skill-gap ──► Direct LLM call (fast)
    └── GET  /api/v1/market/{role} ──► Direct LLM call (fast)

MCP Servers (stdio, each in own process):
  ├── sql_connector    → SQLite/Postgres career DB
  ├── rest_analytics   → Mock / real analytics REST API  
  ├── powerbi_server   → Power BI REST API via MSAL
  └── stat_model_store → scikit-learn PCA/SVM/RF
```

---

## Key LangGraph Concepts

| Concept | What it does |
|---------|-------------|
| `StateGraph` | Defines nodes + edges with typed state |
| `Annotated[List, operator.add]` | Appends messages rather than overwriting |
| `interrupt()` | Pauses graph, surfaces data to human |
| `Command(resume=value)` | Resumes paused graph with human input |
| `MemorySaver` | In-memory checkpointing (use SqliteSaver for persistence) |
| `compile(interrupt_before=[...])` | Breaks before named nodes |

## MCP Protocol Notes
- Each server runs as a separate **stdio** process
- Client connects via `mcp_config.json`
- Tools are discovered via `list_tools()` RPC
- Called via `call_tool(name, arguments)` RPC
- All return `list[TextContent]`
