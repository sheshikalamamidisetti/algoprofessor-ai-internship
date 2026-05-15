"""
MCP Server: REST Analytics API
Exposes career analytics endpoints via MCP
Run: python mcp_servers/rest_analytics/server.py
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

ANALYTICS_BASE_URL = os.getenv("ANALYTICS_API_URL", "http://localhost:8000/analytics")

app = Server("rest-analytics")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_role_trends",
            description="Get hiring trend data for a role over time (monthly data points)",
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "months": {"type": "integer", "default": 12}
                },
                "required": ["role"]
            }
        ),
        Tool(
            name="get_skill_demand",
            description="Get demand ranking of skills in current job market",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["technical", "soft", "domain", "all"]},
                    "top_n": {"type": "integer", "default": 20}
                }
            }
        ),
        Tool(
            name="get_career_path_analytics",
            description="Get common career transitions from a source role",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_role": {"type": "string"},
                    "top_n": {"type": "integer", "default": 5}
                },
                "required": ["from_role"]
            }
        ),
        Tool(
            name="get_compensation_percentiles",
            description="Get salary percentile breakdown for a role in a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "location": {"type": "string", "default": "India"},
                    "experience_years": {"type": "integer"}
                },
                "required": ["role"]
            }
        ),
        Tool(
            name="get_company_analytics",
            description="Get analytics about companies hiring for a specific role",
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "location": {"type": "string"}
                },
                "required": ["role"]
            }
        )
    ]


# ── Mock analytics data (replace with real API calls) ─────────────────────────

def _mock_role_trends(role: str, months: int) -> dict:
    import random
    base = 100
    data = []
    from datetime import datetime, timedelta
    for i in range(months):
        date = (datetime.now() - timedelta(days=30 * (months - i))).strftime("%Y-%m")
        base += random.randint(-5, 15)
        data.append({"month": date, "job_postings": base, "role": role})
    return {"role": role, "trend": "increasing" if data[-1]["job_postings"] > data[0]["job_postings"] else "declining", "data": data}


def _mock_skill_demand(category: str, top_n: int) -> list:
    skills = {
        "technical": ["Python", "SQL", "PyTorch", "Docker", "Kubernetes", "Spark", "dbt", "Airflow"],
        "soft": ["Communication", "Leadership", "Problem Solving", "Stakeholder Management"],
        "domain": ["Finance", "Healthcare AI", "NLP", "Computer Vision", "Time Series"],
    }
    all_skills = []
    cats = list(skills.keys()) if category == "all" else [category]
    for c in cats:
        for i, s in enumerate(skills.get(c, [])):
            all_skills.append({"skill": s, "category": c, "demand_score": round(9.5 - i * 0.3, 1), "yoy_growth": f"{15 - i * 2}%"})
    return sorted(all_skills, key=lambda x: -x["demand_score"])[:top_n]


def _mock_career_paths(from_role: str) -> list:
    paths = {
        "data analyst": [
            {"to_role": "Data Scientist", "transition_rate": 0.35, "avg_months": 18},
            {"to_role": "ML Engineer", "transition_rate": 0.20, "avg_months": 24},
            {"to_role": "Analytics Manager", "transition_rate": 0.25, "avg_months": 30},
        ],
        "ml engineer": [
            {"to_role": "Senior ML Engineer", "transition_rate": 0.45, "avg_months": 24},
            {"to_role": "ML Lead", "transition_rate": 0.30, "avg_months": 36},
        ]
    }
    return paths.get(from_role.lower(), [{"to_role": "Senior " + from_role, "transition_rate": 0.4, "avg_months": 24}])


def _mock_compensation(role: str, location: str, exp: int) -> dict:
    base = {"data analyst": 800000, "data scientist": 1500000, "ml engineer": 2000000}
    b = base.get(role.lower(), 1200000)
    exp_mult = 1 + (exp or 3) * 0.08
    return {
        "role": role, "location": location,
        "p25": int(b * exp_mult * 0.75),
        "p50": int(b * exp_mult),
        "p75": int(b * exp_mult * 1.35),
        "p90": int(b * exp_mult * 1.7),
        "currency": "INR" if location == "India" else "USD"
    }


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "get_role_trends":
        data = _mock_role_trends(arguments["role"], arguments.get("months", 12))
    elif name == "get_skill_demand":
        data = _mock_skill_demand(arguments.get("category", "all"), arguments.get("top_n", 20))
    elif name == "get_career_path_analytics":
        data = _mock_career_paths(arguments["from_role"])
    elif name == "get_compensation_percentiles":
        data = _mock_compensation(arguments["role"], arguments.get("location", "India"), arguments.get("experience_years", 3))
    elif name == "get_company_analytics":
        data = {"role": arguments["role"], "top_companies": ["Google", "Microsoft", "Flipkart", "Swiggy", "Zepto"], "avg_time_to_hire_days": 28}
    else:
        data = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(data, default=str))]


async def main():
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
