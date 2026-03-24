# Day 6: DataAssist Analytics Agent

**Author:** Sheshikala
**Topic:** OpenAI and Claude APIs for data analytics with function calling, Pydantic outputs, and conversational memory

---

## What I Built

A complete DataAssist Analytics Agent that combines OpenAI and Claude APIs with structured
function calling, Pydantic output schemas, conversational memory, time series analysis,
and a full agent that ties all components together for data analytics tasks.

---

## Folder Structure

```
Day_6_Healthcare_Bot/
  openai_client.py     - OpenAI API client for data analytics tasks
  claude_client.py     - Claude API client for data analytics tasks
  data_loader.py       - Titanic dataset loader and preprocessor
  function_calling.py  - Function calling with structured tool definitions
  pydantic_schemas.py  - Pydantic models for structured report outputs
  report_generator.py  - Automated data analysis report generation
  memory_manager.py    - Conversational memory for multi-turn sessions
  ts_analyzer.py       - Time series analysis with LLM-generated insights
  dataassist_agent.py  - Full DataAssist Analytics Agent
  requirements.txt     - Python dependencies
  README.md            - This file
```

---

## How to Run

Install dependencies:
```
pip install -r requirements.txt
```

Run individual files:
```
python data_loader.py
python function_calling.py
python pydantic_schemas.py
python report_generator.py
python memory_manager.py
python ts_analyzer.py
python dataassist_agent.py
```

All files work without API keys using mock fallbacks.

To use real OpenAI API:
```
set OPENAI_API_KEY=your_openai_api_key_here
python openai_client.py
```

To use real Claude API:
```
set ANTHROPIC_API_KEY=your_anthropic_api_key_here
python claude_client.py
```

---

## Key Concepts Learned

| Concept | What I Learned |
|---|---|
| OpenAI API | Chat completions with system and user messages |
| Claude API | Anthropic messages API with structured prompts |
| Function Calling | Define tools as JSON schemas, LLM decides when to call them |
| Pydantic | Validate and structure LLM outputs as typed Python objects |
| Report Generation | Automated data analysis reports from dataset statistics |
| Conversational Memory | Store and retrieve conversation history for context |
| Time Series Analysis | Trend detection, anomaly detection, forecasting with LLM narration |
| DataAssist Agent | Full agent combining all components for data analytics |

---

## Dataset

All files use the Titanic dataset loaded via seaborn.
Time series analyzer uses synthetic ML experiment metrics generated programmatically.
All files fall back to inline data if seaborn is not installed.


