"""
fastapi_streaming.py  ·  Day 47  ·  Apr 23
--------------------------------------------
Goes into: day9_multiagent/  (extended)

FastAPI Server-Sent Events (SSE) for real-time InsightBot responses.
Instead of waiting for the full analysis, results stream token-by-token
to the client — like ChatGPT's streaming interface.

Usage:
    python fastapi_streaming.py          # starts server on port 8000
    python fastapi_streaming.py --demo   # prints demo without starting server
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

# ── Demo mode (no server needed) ─────────────────────────────────────────

def demo_streaming():
    """Shows what SSE streaming looks like without starting the server."""
    print("=" * 60)
    print("Day 47 — FastAPI SSE Streaming Demo")
    print("=" * 60)
    print()
    print("What SSE streaming does:")
    print("  Without SSE: user waits 10-30s for full analysis, then sees it all")
    print("  With SSE:    user sees results token-by-token as they are generated")
    print()
    print("Server endpoint: POST /analyze/stream")
    print("Response type:   text/event-stream")
    print()
    print("Example SSE events streamed to client:")
    print()

    events = [
        ("agent_start",   {"agent": "DataRetriever", "status": "searching knowledge base"}),
        ("token",         {"text": "Found "}),
        ("token",         {"text": "3 relevant "}),
        ("token",         {"text": "reports on churn analysis."}),
        ("agent_start",   {"agent": "StatReasoner", "status": "applying statistical reasoning"}),
        ("token",         {"text": "Logistic regression "}),
        ("token",         {"text": "recommended. "}),
        ("token",         {"text": "Check class balance first."}),
        ("agent_start",   {"agent": "ReportWriter", "status": "generating report"}),
        ("token",         {"text": "# InsightBot Report\n"}),
        ("token",         {"text": "## Executive Summary\n"}),
        ("token",         {"text": "Churn rate: 8.2%..."}),
        ("agent_complete",{"agent": "QualityChecker", "score": 8.5, "status": "complete"}),
        ("done",          {"total_tokens": 450, "latency_ms": 3200}),
    ]

    for event_type, data in events:
        print(f"  event: {event_type}")
        print(f"  data:  {json.dumps(data)}")
        print()

    print("Client-side JavaScript to receive SSE:")
    print()
    print("  const source = new EventSource('/analyze/stream?query=...')")
    print("  source.onmessage = (e) => {")
    print("    const data = JSON.parse(e.data)")
    print("    if (data.text) document.getElementById('output').innerHTML += data.text")
    print("  }")
    print()
    print("To start the real server:")
    print("  python fastapi_streaming.py")
    print("  Then open: http://localhost:8000/docs")


# ── FastAPI SSE Server ─────────────────────────────────────────────────────

def create_app():
    from fastapi import FastAPI, Query
    from fastapi.responses import StreamingResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

    app = FastAPI(
        title="InsightBot Streaming API",
        description="Real-time SSE streaming for multi-agent DS analysis",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def stream_analysis(query: str) -> AsyncGenerator[str, None]:
        """Stream InsightBot analysis as SSE events."""

        agents = [
            ("DataRetriever",  "Searching DS knowledge base..."),
            ("ChartAnalyst",   "Analysing visualisations..."),
            ("StatReasoner",   "Applying statistical reasoning..."),
            ("ReportWriter",   "Writing structured report..."),
            ("QualityChecker", "Evaluating report quality..."),
        ]

        for agent_name, status in agents:
            event = json.dumps({"type": "agent_start",
                                "agent": agent_name, "status": status})
            yield f"event: agent_start\ndata: {event}\n\n"
            await asyncio.sleep(0.1)

            # Simulate token streaming for this agent
            if os.getenv("OPENAI_API_KEY"):
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                system = f"You are the {agent_name} agent in a DS analysis pipeline."
                prompt = f"For the query '{query}', provide your {agent_name} analysis in 2-3 sentences."

                async with client.chat.completions.stream(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=150,
                ) as stream:
                    async for chunk in stream:
                        if chunk.choices[0].delta.content:
                            token_event = json.dumps({
                                "type":  "token",
                                "agent": agent_name,
                                "text":  chunk.choices[0].delta.content,
                            })
                            yield f"data: {token_event}\n\n"
            else:
                # Demo tokens without API
                demo_responses = {
                    "DataRetriever":  f"Found 3 relevant reports for '{query[:30]}'. Key context retrieved.",
                    "ChartAnalyst":   "Revenue trend shows upward growth. Overfitting detected in model chart.",
                    "StatReasoner":   "Recommend logistic regression. Verify class balance with chi-square test.",
                    "ReportWriter":   f"Report generated for '{query[:30]}' with 4 recommendations.",
                    "QualityChecker": "Quality score: 8.5/10. Statistical reasoning sound. Recommendations actionable.",
                }
                words = demo_responses.get(agent_name, "Analysis complete.").split()
                for word in words:
                    token_event = json.dumps({
                        "type":  "token",
                        "agent": agent_name,
                        "text":  word + " ",
                    })
                    yield f"data: {token_event}\n\n"
                    await asyncio.sleep(0.05)

            complete_event = json.dumps({"type": "agent_complete", "agent": agent_name})
            yield f"event: agent_complete\ndata: {complete_event}\n\n"

        done_event = json.dumps({
            "type":      "done",
            "timestamp": datetime.now().isoformat(),
            "query":     query,
        })
        yield f"event: done\ndata: {done_event}\n\n"

    @app.get("/analyze/stream")
    async def analyze_stream(query: str = Query(..., description="DS analysis query")):
        return StreamingResponse(
            stream_analysis(query),
            media_type="text/event-stream",
            headers={
                "Cache-Control":               "no-cache",
                "X-Accel-Buffering":           "no",
                "Access-Control-Allow-Origin": "*",
            },
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "InsightBot Streaming API",
                "timestamp": datetime.now().isoformat()}

    @app.get("/", response_class=HTMLResponse)
    async def ui():
        return """<!DOCTYPE html>
<html>
<head><title>InsightBot Stream</title>
<style>
body{font-family:monospace;max-width:800px;margin:40px auto;padding:0 20px;background:#0f0f1a;color:#e2e8f0}
input{width:100%;padding:10px;background:#1a1a2e;color:#e2e8f0;border:1px solid #7c6ff7;border-radius:6px;font-size:14px}
button{padding:10px 24px;background:#7c6ff7;color:#fff;border:none;border-radius:6px;cursor:pointer;margin-top:8px}
#output{margin-top:20px;padding:16px;background:#1a1a2e;border-radius:8px;min-height:200px;white-space:pre-wrap;font-size:13px}
.agent{color:#7c6ff7;font-weight:bold}
.token{color:#e2e8f0}
</style>
</head>
<body>
<h2>InsightBot — Real-time SSE Streaming</h2>
<input id="q" placeholder="Enter DS query e.g. analyse churn data" value="analyse customer churn patterns"/>
<button onclick="run()">Run Analysis</button>
<div id="output">Output will stream here...</div>
<script>
function run(){
  const q=document.getElementById('q').value;
  const out=document.getElementById('output');
  out.innerHTML='';
  const es=new EventSource('/analyze/stream?query='+encodeURIComponent(q));
  es.onmessage=e=>{
    const d=JSON.parse(e.data);
    if(d.type==='token') out.innerHTML+=d.text;
    if(d.type==='done'){es.close();out.innerHTML+='\n\n[Done]';}
  };
  es.addEventListener('agent_start',e=>{
    const d=JSON.parse(e.data);
    out.innerHTML+='\n\n<span class="agent">['+d.agent+']</span> '+d.status+'\n';
  });
}
</script>
</body>
</html>"""

    return app, uvicorn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 47 — FastAPI SSE Streaming")
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.demo:
        demo_streaming()
    else:
        try:
            app, uvicorn = create_app()
            print(f"InsightBot Streaming API starting on http://localhost:{args.port}")
            print(f"Docs: http://localhost:{args.port}/docs")
            print(f"UI:   http://localhost:{args.port}/")
            uvicorn.run(app, host="0.0.0.0", port=args.port)
        except ImportError:
            print("FastAPI not installed. Run: pip install fastapi uvicorn")
            print("Showing demo instead:\n")
            demo_streaming()
