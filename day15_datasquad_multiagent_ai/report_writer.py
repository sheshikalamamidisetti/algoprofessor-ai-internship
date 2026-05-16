"""
Agent 4: ReportWriter
Responsibility: Compile all agent outputs into a structured HTML analytics report.
"""

import json
from pathlib import Path
from datetime import datetime
from logger import TeamLogger

logger = TeamLogger("ReportWriter")


class ReportWriter:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run(self, plan: dict, eda: dict, ml: dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"datasquad_report_{timestamp}.html"

        html = self._build_html(plan, eda, ml, timestamp)
        report_path.write_text(html, encoding="utf-8")

        # Also save JSON summary
        summary = {
            "generated_at": timestamp,
            "dataset":      plan["dataset_path"],
            "rows":         plan["n_rows"],
            "features":     plan["n_features"],
            "clusters":     ml.get("optimal_k", "N/A"),
            "silhouette":   ml.get("silhouette_score", "N/A"),
            "pca_components": ml["pca"]["n_components"] if not ml.get("skipped") else "N/A",
            "charts":       list(eda.get("charts", {}).keys()),
        }
        json_path = self.output_dir / f"datasquad_summary_{timestamp}.json"
        json_path.write_text(json.dumps(summary, indent=2))

        logger.info(f"HTML Report - > {report_path}")
        logger.info(f"JSON Summary - > {json_path}")
        return str(report_path)

    def _build_html(self, plan, eda, ml, ts):
        skipped = ml.get("skipped", False)
        cluster_rows = ""
        if not skipped:
            for cid, size in enumerate(ml.get("cluster_sizes", [])):
                cluster_rows += f"<tr><td>Cluster {cid}</td><td>{size}</td></tr>"

        pca_info = "" if skipped else f"""
        <div class='card'>
            <h2>🔻 PCA - Dimensionality Reduction</h2>
            <table>
                <tr><th>Components Retained</th><td>{ml['pca']['n_components']}</td></tr>
                <tr><th>Variance Explained</th><td>{ml['pca']['explained_variance_pct']:.1f}%</td></tr>
                <tr><th>Original Features</th><td>{len(ml['clust_cols'])}</td></tr>
            </table>
        </div>
        <div class='card'>
            <h2>🎯 KMeans Clustering</h2>
            <table>
                <tr><th>Optimal K</th><td>{ml['optimal_k']}</td></tr>
                <tr><th>Silhouette Score</th><td>{ml['silhouette_score']:.4f}</td></tr>
            </table>
            <h3>Cluster Sizes</h3>
            <table><tr><th>Cluster</th><th>Count</th></tr>{cluster_rows}</table>
        </div>"""

        desc_rows = ""
        for col, stats in list(eda.get("desc_stats", {}).items())[:6]:
            mn  = stats.get("mean", "-")
            std = stats.get("std",  "-")
            mn_val  = stats.get("min",  "-")
            mx_val  = stats.get("max",  "-")
            desc_rows += f"<tr><td>{col}</td><td>{mn:.2f}</td><td>{std:.2f}</td><td>{mn_val:.2f}</td><td>{mx_val:.2f}</td></tr>"

        chart_imgs = ""
        for name, path in eda.get("charts", {}).items():
            chart_imgs += f"<figure><figcaption>{name}</figcaption><img src='../{path}' alt='{name}'></figure>"

        for name, path in ml.get("charts", {}).items():
            chart_imgs += f"<figure><figcaption>{name}</figcaption><img src='../{path}' alt='{name}'></figure>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DataSquad Analytics Report</title>
<style>
  :root {{
    --bg: #0d0f1a; --surface: #141726; --accent: #6c63ff;
    --accent2: #00d4aa; --text: #e2e8f0; --muted: #8892a4;
    --border: #1e2540;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif;
         padding: 2rem; line-height: 1.6; }}
  header {{ border-bottom: 2px solid var(--accent); padding-bottom: 1rem; margin-bottom: 2rem; }}
  header h1 {{ font-size: 2rem; color: var(--accent); letter-spacing: 2px; }}
  header p  {{ color: var(--muted); font-size: 0.9rem; margin-top: .3rem; }}
  .agents   {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }}
  .agent    {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
               padding: 1rem 1.4rem; flex: 1; min-width: 160px; }}
  .agent .tag {{ font-size: .7rem; color: var(--accent2); text-transform: uppercase;
                 letter-spacing: 1px; margin-bottom: .4rem; }}
  .agent .name {{ font-size: 1rem; font-weight: 600; }}
  .card     {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
               padding: 1.5rem; margin-bottom: 1.5rem; }}
  .card h2  {{ color: var(--accent2); font-size: 1.1rem; margin-bottom: 1rem; }}
  .card h3  {{ color: var(--muted); font-size: .9rem; margin: 1rem 0 .5rem; }}
  table     {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  th, td    {{ padding: .5rem .8rem; border: 1px solid var(--border); text-align: left; }}
  th        {{ background: #1a1f35; color: var(--accent); }}
  .charts   {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
  figure    {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
               padding: .8rem; flex: 1 1 300px; }}
  figcaption{{ font-size: .75rem; color: var(--muted); text-align: center; margin-bottom: .4rem; }}
  figure img{{ width: 100%; border-radius: 6px; }}
  .stat-grid{{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; }}
  .stat-box {{ background: #1a1f35; border-radius: 10px; padding: 1rem; text-align: center; }}
  .stat-box .val {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); }}
  .stat-box .lbl {{ font-size: .75rem; color: var(--muted); margin-top: .2rem; }}
  footer    {{ margin-top: 3rem; color: var(--muted); font-size: .78rem; text-align: center; }}
</style>
</head>
<body>

<header>
  <h1>⬡ DataSquad Analytics Report</h1>
  <p>M10 · 4-Agent Analytics Team · Generated {ts}</p>
  <p style="margin-top:.5rem">Dataset: <strong>{plan['dataset_path']}</strong></p>
</header>

<div class="agents">
  <div class="agent"><div class="tag">Agent 1</div><div class="name">🗺 DataPlanner</div></div>
  <div class="agent"><div class="tag">Agent 2</div><div class="name">📊 StatAnalyst</div></div>
  <div class="agent"><div class="tag">Agent 3</div><div class="name">🤖 MLEngineer</div></div>
  <div class="agent"><div class="tag">Agent 4</div><div class="name">📝 ReportWriter</div></div>
</div>

<div class="card">
  <h2>📁 Dataset Overview</h2>
  <div class="stat-grid">
    <div class="stat-box"><div class="val">{plan['n_rows']:,}</div><div class="lbl">Rows</div></div>
    <div class="stat-box"><div class="val">{plan['n_features']}</div><div class="lbl">Features</div></div>
    <div class="stat-box"><div class="val">{len(plan['numeric_cols'])}</div><div class="lbl">Numeric</div></div>
    <div class="stat-box"><div class="val">{len(plan['categorical_cols'])}</div><div class="lbl">Categorical</div></div>
    <div class="stat-box"><div class="val">{plan['memory_mb']}MB</div><div class="lbl">Memory</div></div>
    <div class="stat-box"><div class="val">{ml.get('optimal_k','-')}</div><div class="lbl">Clusters</div></div>
  </div>
</div>

<div class="card">
  <h2>📈 Statistical Summary (first 6 numeric columns)</h2>
  <table>
    <tr><th>Column</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>
    {desc_rows}
  </table>
</div>

{pca_info}

<div class="card">
  <h2>🖼 EDA & ML Charts</h2>
  <div class="charts">{chart_imgs if chart_imgs else '<p style="color:var(--muted)">Charts saved to outputs/charts/</p>'}</div>
</div>

<footer>DataSquad · M10 Project · NumPy · Pandas · Seaborn · Scikit-learn · Multi-Agent Pipeline</footer>
</body>
</html>"""
