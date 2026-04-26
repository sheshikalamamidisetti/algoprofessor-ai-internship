"""
multimodal_chart_analysis.py  ·  Day 42  ·  Apr 16
----------------------------------------------------
Goes into: day7_rag/  (extended)

GPT-4V and LLaVA for chart/dashboard image analysis.
Extends your Matplotlib/Seaborn and Power BI skills —
now an LLM can read and interpret charts you generate.

Usage:
    python multimodal_chart_analysis.py
    python multimodal_chart_analysis.py --generate-charts
    python multimodal_chart_analysis.py --analyze outputs/chart.png
"""

import argparse
import base64
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Chart Generation (extends Matplotlib/Seaborn from Phase 1) ────────────

def generate_sample_charts(out_dir: str = "outputs") -> list[str]:
    """Generate sample DS charts to analyze — extends Phase 1 Matplotlib skills."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams["figure.facecolor"] = "white"
    import numpy as np
    import pandas as pd

    Path(out_dir).mkdir(exist_ok=True)
    paths = []

    # Chart 1 — Time series with trend
    fig, ax = plt.subplots(figsize=(10, 5))
    months = pd.date_range("2023-01", periods=24, freq="M")
    values = [100 + i * 8 + np.random.normal(0, 10) for i in range(24)]
    ax.plot(months, values, "b-o", markersize=4, label="Monthly Revenue ($k)")
    z = np.polyfit(range(24), values, 1)
    trend = np.poly1d(z)(range(24))
    ax.plot(months, trend, "r--", label="Trend line", linewidth=2)
    ax.set_title("Monthly Revenue 2023–2024 with Trend", fontsize=14)
    ax.set_ylabel("Revenue ($k)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = f"{out_dir}/timeseries_chart.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(path1)
    print(f"Generated: {path1}")

    # Chart 2 — Model comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = ["Random Forest", "SVM", "Gradient Boost", "Neural Net", "Logistic Reg"]
    train_acc = [0.97, 0.91, 0.95, 0.93, 0.85]
    test_acc  = [0.74, 0.88, 0.85, 0.80, 0.83]
    x = range(len(models))
    axes[0].bar([i - 0.2 for i in x], train_acc, 0.4, label="Train", color="#4C72B0")
    axes[0].bar([i + 0.2 for i in x], test_acc,  0.4, label="Test",  color="#DD8452")
    axes[0].set_title("Model Comparison: Train vs Test Accuracy")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.6, 1.0)
    axes[0].axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="90% threshold")
    axes[0].legend()

    # Correlation heatmap
    import seaborn as sns
    np.random.seed(42)
    corr_data = pd.DataFrame(
        np.random.randn(100, 5),
        columns=["Revenue", "Churn", "Tenure", "Usage", "Support"],
    ).corr()
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1], center=0, square=True)
    axes[1].set_title("Feature Correlation Matrix")

    plt.tight_layout()
    path2 = f"{out_dir}/model_comparison.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(path2)
    print(f"Generated: {path2}")

    return paths


# ── GPT-4V Image Analysis ──────────────────────────────────────────────────

def encode_image(image_path: str) -> str:
    """Convert image to base64 for GPT-4V."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_with_gpt4v(image_path: str, question: str = None) -> dict:
    """Analyze chart using GPT-4V vision."""
    from openai import OpenAI
    import time

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    b64 = encode_image(image_path)

    prompt = question or (
        "You are a senior data scientist analyzing this chart. "
        "Provide: 1) What the chart shows, 2) Key findings and patterns, "
        "3) Any anomalies or concerns, 4) Specific actionable recommendations."
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}",
                                   "detail": "high"}},
                ],
            }
        ],
        max_tokens=800,
    )
    return {
        "model": "gpt-4o-vision",
        "image": image_path,
        "analysis": resp.choices[0].message.content,
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "tokens": resp.usage.total_tokens,
    }


def analyze_with_llava(image_path: str, question: str = None) -> dict:
    """Analyze chart using LLaVA via Ollama (local, free)."""
    import ollama
    import base64
    import time

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    prompt = question or (
        "Analyze this data science chart. Describe: "
        "1) What is shown, 2) Key trends or patterns, 3) Any issues, 4) Recommendations."
    )

    t0 = time.time()
    resp = ollama.chat(
        model="llava",
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_bytes],
        }],
    )
    return {
        "model": "llava (local)",
        "image": image_path,
        "analysis": resp["message"]["content"],
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


def analyze_chart(image_path: str, use_local: bool = False):
    """Analyze a chart with GPT-4V or LLaVA."""
    print(f"\nAnalyzing: {image_path}")
    print("=" * 70)

    if use_local:
        result = analyze_with_llava(image_path)
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print("No OPENAI_API_KEY — showing demo analysis")
            result = {
                "model": "demo",
                "image": image_path,
                "analysis": (
                    "DEMO OUTPUT — Chart Analysis:\n\n"
                    "1) WHAT THE CHART SHOWS: Monthly revenue from Jan 2023 to Dec 2024 "
                    "with a trend line overlay showing consistent upward growth.\n\n"
                    "2) KEY FINDINGS: Revenue grew from approximately $100k to $292k "
                    "over 24 months — a 192% increase. The trend line shows strong "
                    "linear growth of ~$8k per month.\n\n"
                    "3) ANOMALIES: Month 6 shows a dip below trend suggesting a one-time "
                    "disruption. Month 18 shows a spike above trend.\n\n"
                    "4) RECOMMENDATIONS: Investigate month 6 dip. Forecast next 3 months "
                    "using the fitted trend. Consider seasonal decomposition."
                ),
                "latency_ms": 0,
            }
        else:
            result = analyze_with_gpt4v(image_path)

    print(f"Model: {result['model']} | Latency: {result['latency_ms']}ms")
    print(f"\nAnalysis:\n{result['analysis']}")

    # Save result
    out_path = image_path.replace(".png", "_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 42 — Multimodal Chart Analysis")
    parser.add_argument("--generate-charts", action="store_true")
    parser.add_argument("--analyze", default=None, help="Path to chart image")
    parser.add_argument("--local", action="store_true", help="Use LLaVA instead of GPT-4V")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    if args.generate_charts or args.analyze is None:
        print("Generating sample charts...")
        chart_paths = generate_sample_charts()
        for path in chart_paths:
            analyze_chart(path, use_local=args.local)
    else:
        analyze_chart(args.analyze, use_local=args.local)
