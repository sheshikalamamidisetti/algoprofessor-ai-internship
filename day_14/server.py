"""
MCP Server: SQL Data Connector
Extends SQL skills — exposes career/talent DB via MCP protocol
Run: python mcp_servers/sql_connector/server.py
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./talentTwin.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

app = Server("sql-connector")


# ── Schema setup (run once) ───────────────────────────────────────────────────

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS professionals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    current_role TEXT,
    years_exp INTEGER,
    skills TEXT,  -- JSON array
    location TEXT,
    education TEXT,
    target_role TEXT
);

CREATE TABLE IF NOT EXISTS job_market (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    location TEXT,
    avg_salary_inr INTEGER,
    demand_score REAL,
    top_skills TEXT,  -- JSON array
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS skill_taxonomy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill TEXT NOT NULL,
    category TEXT,
    related_roles TEXT,  -- JSON array
    learning_resources TEXT  -- JSON array
);
"""


async def init_db():
    async with engine.begin() as conn:
        for stmt in CREATE_TABLES_SQL.strip().split(";"):
            if stmt.strip():
                await conn.execute(sa.text(stmt))


# ── MCP Tool definitions ──────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="sql_query",
            description="Execute a read-only SQL SELECT query on the TalentTwin database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT query"},
                    "params": {"type": "object", "description": "Named bind params", "default": {}}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_professionals",
            description="Retrieve professionals filtered by role, skills, or location",
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "skill": {"type": "string"},
                    "location": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="get_job_market",
            description="Get job market data: demand scores and salary ranges by role",
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "location": {"type": "string"}
                }
            }
        ),
        Tool(
            name="get_skill_taxonomy",
            description="Look up a skill: category, related roles, learning resources",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill": {"type": "string", "description": "Skill name to look up"}
                },
                "required": ["skill"]
            }
        ),
        Tool(
            name="upsert_professional",
            description="Insert or update a professional profile in the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "current_role": {"type": "string"},
                    "years_exp": {"type": "integer"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "location": {"type": "string"},
                    "target_role": {"type": "string"}
                },
                "required": ["name"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    async with AsyncSessionLocal() as session:
        if name == "sql_query":
            query = arguments["query"]
            if not query.strip().upper().startswith("SELECT"):
                return [TextContent(type="text", text="Error: only SELECT queries allowed")]
            result = await session.execute(sa.text(query), arguments.get("params", {}))
            rows = [dict(r._mapping) for r in result]
            return [TextContent(type="text", text=json.dumps(rows, default=str))]

        elif name == "get_professionals":
            conditions, params = [], {}
            if role := arguments.get("role"):
                conditions.append("current_role LIKE :role")
                params["role"] = f"%{role}%"
            if skill := arguments.get("skill"):
                conditions.append("skills LIKE :skill")
                params["skill"] = f"%{skill}%"
            if location := arguments.get("location"):
                conditions.append("location LIKE :loc")
                params["loc"] = f"%{location}%"
            where = "WHERE " + " AND ".join(conditions) if conditions else ""
            limit = arguments.get("limit", 10)
            q = f"SELECT * FROM professionals {where} LIMIT {limit}"
            result = await session.execute(sa.text(q), params)
            rows = [dict(r._mapping) for r in result]
            return [TextContent(type="text", text=json.dumps(rows, default=str))]

        elif name == "get_job_market":
            conditions, params = [], {}
            if role := arguments.get("role"):
                conditions.append("role LIKE :role")
                params["role"] = f"%{role}%"
            if loc := arguments.get("location"):
                conditions.append("location LIKE :loc")
                params["loc"] = f"%{loc}%"
            where = "WHERE " + " AND ".join(conditions) if conditions else ""
            result = await session.execute(sa.text(f"SELECT * FROM job_market {where}"), params)
            rows = [dict(r._mapping) for r in result]
            return [TextContent(type="text", text=json.dumps(rows, default=str))]

        elif name == "get_skill_taxonomy":
            skill = arguments["skill"]
            result = await session.execute(
                sa.text("SELECT * FROM skill_taxonomy WHERE skill LIKE :s"),
                {"s": f"%{skill}%"}
            )
            rows = [dict(r._mapping) for r in result]
            return [TextContent(type="text", text=json.dumps(rows, default=str))]

        elif name == "upsert_professional":
            args = arguments.copy()
            args["skills"] = json.dumps(args.get("skills", []))
            await session.execute(
                sa.text("""INSERT INTO professionals (name, current_role, years_exp, skills, location, target_role)
                           VALUES (:name, :current_role, :years_exp, :skills, :location, :target_role)"""),
                args
            )
            await session.commit()
            return [TextContent(type="text", text="Professional profile saved successfully")]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    await init_db()
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
