"""
TalentTwin FastAPI Backend
Run: uvicorn talentTwin.api.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import uuid
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="TalentTwin API",
    description="AI Career Digital Twin — Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store (replace with Redis/DB in prod) ───────────────────────
jobs: dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ProfileRequest(BaseModel):
    name: str
    current_role: str
    years_experience: int = Field(ge=0, le=50)
    skills: List[str]
    location: str = "India"
    target_role: str
    education: Optional[str] = None

class RoadmapRequest(BaseModel):
    profile: ProfileRequest
    duration_months: int = Field(default=6, ge=1, le=24)
    focus: Optional[str] = None  # "technical" | "leadership" | "balanced"

class SkillGapRequest(BaseModel):
    current_skills: List[str]
    target_role: str
    location: str = "India"

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | done | failed
    result: Optional[dict] = None
    created_at: str
    completed_at: Optional[str] = None


# ── Background task runner ────────────────────────────────────────────────────

async def _run_langgraph_flow(job_id: str, profile: ProfileRequest, duration_months: int, focus: Optional[str]):
    """Run the full LangGraph TalentTwin pipeline async."""
    jobs[job_id]["status"] = "running"
    try:
        # Import graph here to avoid circular imports
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from langgraph.state_machines.talent_graph import build_talent_graph, TalentTwinState
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.messages import HumanMessage

        user_msg = (
            f"My name is {profile.name}. I'm a {profile.current_role} with "
            f"{profile.years_experience} years of experience. "
            f"My skills: {', '.join(profile.skills)}. "
            f"Location: {profile.location}. "
            f"Target: {profile.target_role}. "
            f"Create a {duration_months}-month roadmap."
        )
        if focus:
            user_msg += f" Focus: {focus}."

        graph = build_talent_graph()
        compiled = graph.compile(checkpointer=MemorySaver())
        initial_state: TalentTwinState = {
            "messages": [HumanMessage(content=user_msg)],
            "user_profile": {}, "skills_identified": [],
            "target_role": profile.target_role,
            "market_data": {}, "skill_gaps": [], "roadmap": "",
            "qa_approved": False, "qa_feedback": "",
            "current_node": "", "iteration": 0
        }
        config = {"configurable": {"thread_id": job_id}}
        result = compiled.invoke(initial_state, config=config)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "roadmap": result["roadmap"],
            "skill_gaps": result["skill_gaps"],
            "market_data": result["market_data"],
            "qa_approved": result["qa_approved"],
            "qa_feedback": result["qa_feedback"]
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = {"error": str(e)}
    finally:
        jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "TalentTwin API", "version": "1.0.0", "status": "healthy"}


@app.post("/api/v1/roadmap", response_model=JobStatus, status_code=202)
async def create_roadmap(req: RoadmapRequest, background_tasks: BackgroundTasks):
    """Kick off async roadmap generation. Poll /jobs/{job_id} for results."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "created_at": datetime.utcnow().isoformat(), "result": None, "completed_at": None}
    background_tasks.add_task(_run_langgraph_flow, job_id, req.profile, req.duration_months, req.focus)
    return JobStatus(job_id=job_id, status="pending", created_at=jobs[job_id]["created_at"])


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = jobs[job_id]
    return JobStatus(job_id=job_id, **j)


@app.post("/api/v1/skill-gap")
async def skill_gap(req: SkillGapRequest):
    """Synchronous skill gap analysis (fast, no LangGraph needed)."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    import json

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    response = llm.invoke([
        SystemMessage(content="Return JSON only: {\"required_skills\": [...], \"gaps\": [...], \"coverage_pct\": N}"),
        HumanMessage(content=f"Current skills: {req.current_skills}\nTarget role: {req.target_role}\nLocation: {req.location}")
    ])
    try:
        return json.loads(response.content)
    except Exception:
        return {"raw": response.content}


@app.get("/api/v1/market/{role}")
async def market_data(role: str, location: str = "India"):
    """Quick market data lookup."""
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    import json

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    response = llm.invoke([
        SystemMessage(content='Return JSON: {"demand": "high/medium/low", "avg_salary_inr": N, "top_skills": [...], "growth": "X%"}'),
        HumanMessage(content=f"Role: {role}, Location: {location}")
    ])
    try:
        return json.loads(response.content)
    except Exception:
        return {"raw": response.content}


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "jobs_in_memory": len(jobs)}
