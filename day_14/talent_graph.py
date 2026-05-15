"""
Day 64 — LangGraph State Machine: TalentTwin Career Assessment Flow

States:
  INTAKE → PROFILE_BUILD → SKILL_ANALYSIS → MARKET_RESEARCH
         → GAP_ANALYSIS → ROADMAP_GENERATION → REVIEW → DONE
"""
from __future__ import annotations

from typing import TypedDict, Annotated, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from dotenv import load_dotenv
import operator

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)


# ── State schema ──────────────────────────────────────────────────────────────

class TalentTwinState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_profile: dict
    skills_identified: List[str]
    target_role: str
    market_data: dict
    skill_gaps: List[str]
    roadmap: str
    qa_approved: bool
    qa_feedback: str
    current_node: str
    iteration: int


# ── Node functions ────────────────────────────────────────────────────────────

def intake_node(state: TalentTwinState) -> TalentTwinState:
    """Extract user intent and basic profile from initial messages."""
    messages = state["messages"]
    response = llm.invoke([
        SystemMessage(content="""Extract from the user's message:
1. Current job title / role
2. Years of experience  
3. Target role they want to reach
4. Current skills mentioned
Return as JSON: {"current_role": "...", "years_exp": N, "target_role": "...", "skills": [...]}"""),
        *messages
    ])
    import json
    try:
        profile = json.loads(response.content)
    except Exception:
        profile = {"current_role": "unknown", "years_exp": 0, "target_role": "unknown", "skills": []}
    
    return {
        **state,
        "user_profile": profile,
        "current_node": "intake",
        "iteration": state.get("iteration", 0) + 1
    }


def profile_build_node(state: TalentTwinState) -> TalentTwinState:
    """Enrich user profile with inferred details."""
    profile = state["user_profile"]
    response = llm.invoke([
        SystemMessage(content="You are a career profiling expert. Enrich the profile with education guess, industry, and seniority level."),
        HumanMessage(content=f"Profile so far: {profile}")
    ])
    enriched = {**profile, "enrichment": response.content}
    return {**state, "user_profile": enriched, "current_node": "profile_build"}


def skill_analysis_node(state: TalentTwinState) -> TalentTwinState:
    """Identify and categorise skills from profile."""
    import json
    response = llm.invoke([
        SystemMessage(content="""Given the user profile, list all skills. 
Return JSON: {"technical": [...], "soft": [...], "domain": [...]}"""),
        HumanMessage(content=str(state["user_profile"]))
    ])
    try:
        skills_data = json.loads(response.content)
        all_skills = (
            skills_data.get("technical", []) +
            skills_data.get("soft", []) +
            skills_data.get("domain", [])
        )
    except Exception:
        all_skills = state["user_profile"].get("skills", [])
    
    return {**state, "skills_identified": all_skills, "current_node": "skill_analysis"}


def market_research_node(state: TalentTwinState) -> TalentTwinState:
    """Research market demand for target role."""
    import json
    target = state["user_profile"].get("target_role", "Data Scientist")
    response = llm.invoke([
        SystemMessage(content="You are a job market analyst. Provide market data for the role."),
        HumanMessage(content=f"""Target role: {target}
Provide JSON with: {{"demand": "high/medium/low", "avg_salary_inr": N, 
"top_skills_required": [...], "top_hiring_companies": [...], "yoy_growth": "X%"}}""")
    ])
    try:
        market = json.loads(response.content)
    except Exception:
        market = {"demand": "high", "avg_salary_inr": 1500000, "top_skills_required": []}
    
    return {**state, "market_data": market, "current_node": "market_research"}


def gap_analysis_node(state: TalentTwinState) -> TalentTwinState:
    """Calculate skill gaps between current skills and market requirements."""
    import json
    current = set(s.lower() for s in state["skills_identified"])
    required = set(s.lower() for s in state["market_data"].get("top_skills_required", []))
    gaps = list(required - current)
    return {**state, "skill_gaps": gaps, "current_node": "gap_analysis"}


def roadmap_generation_node(state: TalentTwinState) -> TalentTwinState:
    """Generate a personalised learning roadmap."""
    response = llm.invoke([
        SystemMessage(content="You are a career coach. Create a detailed, actionable roadmap."),
        HumanMessage(content=f"""
User profile: {state['user_profile']}
Current skills: {state['skills_identified']}
Skill gaps: {state['skill_gaps']}
Target role: {state['user_profile'].get('target_role')}
Market data: {state['market_data']}

Create a 6-month weekly roadmap with specific courses, projects, and milestones.
""")
    ])
    return {**state, "roadmap": response.content, "current_node": "roadmap_generation"}


def qa_review_node(state: TalentTwinState) -> TalentTwinState:
    """Auto-QA the roadmap for completeness."""
    import json
    response = llm.invoke([
        SystemMessage(content="""Review this career roadmap for quality. 
Return JSON: {"approved": true/false, "score": 0-10, "issues": [...], "feedback": "..."}"""),
        HumanMessage(content=f"Roadmap to review:\n{state['roadmap']}")
    ])
    try:
        qa = json.loads(response.content)
        approved = qa.get("approved", True) and qa.get("score", 0) >= 7
        feedback = qa.get("feedback", "")
    except Exception:
        approved, feedback = True, "Auto-approved"
    
    return {
        **state,
        "qa_approved": approved,
        "qa_feedback": feedback,
        "current_node": "qa_review"
    }


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_after_qa(state: TalentTwinState) -> str:
    if state.get("qa_approved"):
        return "done"
    if state.get("iteration", 0) >= 3:
        return "done"  # max retries reached
    return "roadmap_generation"  # retry with feedback


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_talent_graph() -> StateGraph:
    graph = StateGraph(TalentTwinState)

    graph.add_node("intake", intake_node)
    graph.add_node("profile_build", profile_build_node)
    graph.add_node("skill_analysis", skill_analysis_node)
    graph.add_node("market_research", market_research_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("roadmap_generation", roadmap_generation_node)
    graph.add_node("qa_review", qa_review_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "profile_build")
    graph.add_edge("profile_build", "skill_analysis")
    graph.add_edge("skill_analysis", "market_research")
    graph.add_edge("market_research", "gap_analysis")
    graph.add_edge("gap_analysis", "roadmap_generation")
    graph.add_edge("roadmap_generation", "qa_review")
    graph.add_conditional_edges("qa_review", route_after_qa, {"done": END, "roadmap_generation": "roadmap_generation"})

    return graph


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    graph = build_talent_graph()
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    initial_state: TalentTwinState = {
        "messages": [HumanMessage(content=(
            "I'm a data analyst with 3 years of experience. "
            "I know Python, SQL, Excel, and Tableau. "
            "I want to become an ML Engineer. Help me!"
        ))],
        "user_profile": {},
        "skills_identified": [],
        "target_role": "ML Engineer",
        "market_data": {},
        "skill_gaps": [],
        "roadmap": "",
        "qa_approved": False,
        "qa_feedback": "",
        "current_node": "",
        "iteration": 0
    }

    config = {"configurable": {"thread_id": "demo-001"}}
    result = compiled.invoke(initial_state, config=config)

    print("\n FINAL ROADMAP:")
    print(result["roadmap"])
    print(f"\n QA Approved: {result['qa_approved']}")
    print(f" QA Feedback: {result['qa_feedback']}")
