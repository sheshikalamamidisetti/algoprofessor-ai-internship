"""
MCP Server: Power BI Report Server
Extends BI knowledge — fetch, embed, refresh Power BI reports via MCP
Run: python mcp_servers/powerbi_server/server.py
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional
import msal
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from dotenv import load_dotenv
import os

load_dotenv()

TENANT_ID    = os.getenv("POWERBI_TENANT_ID", "")
CLIENT_ID    = os.getenv("POWERBI_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("POWERBI_CLIENT_SECRET", "")
WORKSPACE_ID  = os.getenv("POWERBI_WORKSPACE_ID", "")
SCOPE = ["https://analysis.windows.net/powerbi/api/.default"]
PBI_BASE = "https://api.powerbi.com/v1.0/myorg"

app = Server("powerbi-server")
_token_cache: dict = {}


def get_access_token() -> str:
    """Acquire Power BI access token via MSAL client credentials."""
    if not all([TENANT_ID, CLIENT_ID, CLIENT_SECRET]):
        return "DEMO_TOKEN_NO_CREDENTIALS"
    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    pbi_app = msal.ConfidentialClientApplication(CLIENT_ID, CLIENT_SECRET, authority=authority)
    result = pbi_app.acquire_token_silent(SCOPE, account=None)
    if not result:
        result = pbi_app.acquire_token_for_client(SCOPE)
    return result.get("access_token", "")


async def pbi_get(path: str) -> dict:
    token = get_access_token()
    if token == "DEMO_TOKEN_NO_CREDENTIALS":
        return {"demo": True, "message": "Set Power BI env vars to use real data"}
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{PBI_BASE}{path}", headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        return r.json()


async def pbi_post(path: str, body: dict) -> dict:
    token = get_access_token()
    if token == "DEMO_TOKEN_NO_CREDENTIALS":
        return {"demo": True, "message": "Set Power BI env vars to use real data"}
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{PBI_BASE}{path}", headers={"Authorization": f"Bearer {token}"}, json=body)
        r.raise_for_status()
        return r.json() if r.text else {"status": "ok"}


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_reports",
            description="List all Power BI reports in the configured workspace",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_report_embed_token",
            description="Get an embed token for a specific Power BI report (for embedding in web apps)",
            inputSchema={
                "type": "object",
                "properties": {
                    "report_id": {"type": "string", "description": "Power BI report GUID"}
                },
                "required": ["report_id"]
            }
        ),
        Tool(
            name="list_datasets",
            description="List all datasets in the workspace",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="refresh_dataset",
            description="Trigger a dataset refresh in Power BI",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"}
                },
                "required": ["dataset_id"]
            }
        ),
        Tool(
            name="get_report_pages",
            description="List all pages (tabs) within a Power BI report",
            inputSchema={
                "type": "object",
                "properties": {
                    "report_id": {"type": "string"}
                },
                "required": ["report_id"]
            }
        ),
        Tool(
            name="execute_dax_query",
            description="Run a DAX query against a Power BI dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "dax_query": {"type": "string"}
                },
                "required": ["dataset_id", "dax_query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "list_reports":
            ws = f"/groups/{WORKSPACE_ID}" if WORKSPACE_ID else ""
            data = await pbi_get(f"{ws}/reports")

        elif name == "get_report_embed_token":
            report_id = arguments["report_id"]
            ws = f"/groups/{WORKSPACE_ID}" if WORKSPACE_ID else ""
            data = await pbi_post(
                f"{ws}/reports/{report_id}/GenerateToken",
                {"accessLevel": "View"}
            )

        elif name == "list_datasets":
            ws = f"/groups/{WORKSPACE_ID}" if WORKSPACE_ID else ""
            data = await pbi_get(f"{ws}/datasets")

        elif name == "refresh_dataset":
            ws = f"/groups/{WORKSPACE_ID}" if WORKSPACE_ID else ""
            data = await pbi_post(f"{ws}/datasets/{arguments['dataset_id']}/refreshes", {})

        elif name == "get_report_pages":
            ws = f"/groups/{WORKSPACE_ID}" if WORKSPACE_ID else ""
            data = await pbi_get(f"{ws}/reports/{arguments['report_id']}/pages")

        elif name == "execute_dax_query":
            ws = f"/groups/{WORKSPACE_ID}" if WORKSPACE_ID else ""
            data = await pbi_post(
                f"{ws}/datasets/{arguments['dataset_id']}/executeQueries",
                {"queries": [{"query": arguments["dax_query"]}], "serializerSettings": {"includeNulls": True}}
            )

        else:
            data = {"error": f"Unknown tool: {name}"}

    except Exception as e:
        data = {"error": str(e)}

    return [TextContent(type="text", text=json.dumps(data, default=str))]


async def main():
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
