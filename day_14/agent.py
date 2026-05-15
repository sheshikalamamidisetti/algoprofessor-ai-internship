"""
Day 61 — ReAct Agent (Reasoning + Acting)
Pattern: Think → Act → Observe → Repeat until done
"""
from __future__ import annotations

import json
from typing import Any
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

# ── Tools available to the ReAct agent ──────────────────────────────────────

@tool
def get_career_skills(job_title: str) -> str:
    """Fetch the top skills required for a given job title from the skills DB."""
    # Placeholder — wire to your SQL MCP server
    skills_db = {
        "data scientist": ["Python", "ML", "SQL", "Statistics", "Communication"],
        "ml engineer": ["Python", "MLOps", "Docker", "Kubernetes", "PyTorch"],
        "data analyst": ["SQL", "Excel", "Tableau", "Python", "Statistics"],
    }
    return json.dumps(skills_db.get(job_title.lower(), ["No data found"]))


@tool
def get_salary_range(job_title: str, location: str = "India") -> str:
    """Return typical salary range for a role in a given location."""
    ranges = {
        "data scientist": {"India": "₹12L–₹35L", "US": "$90k–$160k"},
        "ml engineer": {"India": "₹15L–₹45L", "US": "$110k–$200k"},
    }
    data = ranges.get(job_title.lower(), {})
    return json.dumps(data.get(location, "Data not available"))


@tool
def skill_gap_analysis(current_skills: str, target_role: str) -> str:
    """Given current skills (comma-separated), return missing skills for target role."""
    role_skills = {
        "ml engineer": {"Python", "MLOps", "Docker", "Kubernetes", "PyTorch"},
        "data scientist": {"Python", "ML", "SQL", "Statistics", "Communication"},
    }
    current = {s.strip().lower() for s in current_skills.split(",")}
    required = {s.lower() for s in role_skills.get(target_role.lower(), set())}
    gap = required - current
    return json.dumps({"missing_skills": list(gap), "coverage": f"{len(required - gap)}/{len(required)}"})


# ── Prompt ───────────────────────────────────────────────────────────────────

REACT_PROMPT = PromptTemplate.from_template("""
You are TalentTwin, an AI career advisor. Use the available tools to help users 
understand their career path, skill gaps, and growth opportunities.

You have access to these tools:
{tools}

Use this format STRICTLY:
Question: the input question you must answer
Thought: reason about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!
Question: {input}
Thought: {agent_scratchpad}
""")


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_react_agent(verbose: bool = True) -> AgentExecutor:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    tools = [get_career_skills, get_salary_range, skill_gap_analysis, DuckDuckGoSearchRun()]
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose, max_iterations=10)


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    executor = build_react_agent()
    result = executor.invoke({
        "input": (
            "I know Python, SQL, and Excel. "
            "What skills am I missing to become an ML Engineer, "
            "and what salary can I expect in India?"
        )
    })
    print("\n Final Answer:", result["output"])
