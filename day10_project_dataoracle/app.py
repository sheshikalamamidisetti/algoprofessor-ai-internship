"""
app.py  ·  DataOracle Capstone
-------------------------------
Integrates all Day 31-35 components into one Gradio app.

    Day 31 → model_registry.py      (LLM wrappers)
    Day 32 → benchmark_runner.py    (DS benchmarks)
    Day 33 → tot_dspy_pipeline.py   (ToT reasoning)
    Day 34 → pydantic_schemas.py    (structured outputs)
    Day 35 → ml_report_generator.py (automated report)
         ↓
    app.py → full Gradio UI

Run:
    python app.py
    open http://localhost:7860
"""

import os, json
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from benchmark_runner import run, save, TASKS
from tot_dspy_pipeline import run_tot, PROBLEMS
from ml_report_generator import build_report, save_report
from pydantic_schemas import LLMBenchmarkEntry


# ── Tab handlers ───────────────────────────────────────────────────────────

def bench_handler(models_str: str, cats_str: str) -> tuple[str, str]:
    models = [m.strip() for m in models_str.split(",") if m.strip()]
    cats   = [c.strip() for c in cats_str.split(",")   if c.strip()]
    results = run(models, cats)
    if not results:
        return "No results — check API keys in .env", ""
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r["score_pct"])
    rows = ["| Model | Avg Score % | Tasks |",
            "|-------|------------|-------|"]
    for m, s in sorted(by_model.items(), key=lambda x: -sum(x[1])/len(x[1])):
        rows.append(f"| {m} | {sum(s)/len(s):.1f}% | {len(s)} |")
    path = save(results)
    return "\n".join(rows), f"Saved → {path}"


def tot_handler(problem_key: str) -> str:
    try:
        r = run_tot(problem_key)
        return (f"**Branches explored:** {r['branches_explored']}\n\n"
                f"**Best branch:** {r['best_branch']}\n\n"
                f"**Justification:** {r['justification']}\n\n"
                f"**Final Answer:**\n\n{r['final_answer']}")
    except Exception as e:
        return f"Error: {e}\n\nCheck OPENAI_API_KEY in .env"


def report_handler(json_path: str) -> tuple[str, str]:
    path = json_path.strip() if json_path else None
    report = build_report(path)
    md_path = save_report(report)
    with open(md_path) as f:
        content = f.read()
    return content, f"Saved → {md_path}"


# ── Gradio UI ──────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="DataOracle", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
# DataOracle — LLM Insights Predictor
**Phase 2 Milestone | Sheshikala | IIT Indore AI & DS Programme**

Built across Days 31–35 → integrated capstone in `day10_project/`
        """)

        with gr.Tabs():

            with gr.Tab("Day 32 — LLM Benchmark"):
                gr.Markdown("Run DS benchmark tasks across GPT-4o, Claude, Gemini, Llama")
                with gr.Row():
                    m_in = gr.Textbox(label="Models (comma-separated)",
                                      value="gpt4o, claude")
                    c_in = gr.Textbox(label="Task categories",
                                      value="stat_inference, ml_code, eda")
                gr.Button("Run Benchmark", variant="primary").click(
                    bench_handler, [m_in, c_in],
                    [gr.Markdown(label="Results"), gr.Textbox(label="Status")])

            with gr.Tab("Day 33 — ToT Reasoning"):
                gr.Markdown("Tree-of-Thought + DSPy statistical reasoning")
                p_dd = gr.Dropdown(choices=list(PROBLEMS.keys()),
                                   value="ttest", label="Problem")
                gr.Button("Run ToT", variant="primary").click(
                    tot_handler, [p_dd], [gr.Markdown(label="Reasoning")])

            with gr.Tab("Day 35 — Generate Report"):
                gr.Markdown("Generate the Milestone 6 DataOracle Report")
                j_in = gr.Textbox(
                    label="Benchmark JSON path (blank = demo report)",
                    placeholder="outputs/benchmark_20240401_120000.json")
                gr.Button("Generate Report", variant="primary").click(
                    report_handler, [j_in],
                    [gr.Markdown(label="Report Preview"),
                     gr.Textbox(label="Saved to")])

            with gr.Tab("About"):
                gr.Markdown("""
### Architecture

```
Day 31  model_registry.py    → GPT-4o · Claude · Gemini · Llama (unified .chat())
Day 32  benchmark_runner.py  → stat_inference · ml_code · eda tasks → CSV + JSON
Day 33  tot_dspy_pipeline.py → 3 branches → best statistical approach
Day 34  pydantic_schemas.py  → HypothesisTestResult · MLModelEval · DataOracleReport
Day 35  ml_report_generator  → LLM insights + Pydantic validation → MD + JSON + PDF
         ↓
app.py                       → Gradio UI integrating all 5 components
```

**Student:** Sheshikala · **Milestone:** M6 DataOracle · **Phase:** 2
                """)

    return app


if __name__ == "__main__":
    build_app().launch(share=False, server_port=7860)
