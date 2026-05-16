"""
AutoEDAPipeline - Extends NumPy/Pandas/Seaborn stack
Used by StatAnalyst agent. Generates charts + outlier reports automatically.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from logger import TeamLogger

logger = TeamLogger("AutoEDA")

CHART_DIR = Path("outputs/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "viridis"
STYLE   = "darkgrid"


class AutoEDAPipeline:
    """
    Automated EDA pipeline. Runs: distributions, correlation heatmap,
    boxplots, categorical counts, pair-plots (small datasets).
    """

    def run(self, df: pd.DataFrame, num_cols: list, cat_cols: list) -> dict:
        sns.set_style(STYLE)
        plt.rcParams.update({"figure.facecolor": "#0d0f1a", "axes.facecolor": "#141726",
                              "text.color": "#e2e8f0", "axes.labelcolor": "#e2e8f0",
                              "xtick.color": "#8892a4", "ytick.color": "#8892a4"})

        charts = {}
        outlier_counts = {}
        top_corrs = []
        value_counts_dict = {}

        # 1. Distributions
        if num_cols:
            fig, axes = plt.subplots(
                max(1, (len(num_cols) + 2) // 3), min(3, len(num_cols)),
                figsize=(15, max(4, 4 * ((len(num_cols) + 2) // 3)))
            )
            axes = np.array(axes).flatten()
            for i, col in enumerate(num_cols[:9]):
                sns.histplot(df[col], ax=axes[i], kde=True, color="#6c63ff")
                axes[i].set_title(col, color="#e2e8f0")
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            p = CHART_DIR / "01_distributions.png"
            plt.savefig(p, dpi=100, bbox_inches="tight", facecolor="#0d0f1a")
            plt.close()
            charts["Distributions"] = str(p)
            logger.info("Chart saved: distributions")

        # 2. Correlation Heatmap
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(min(14, len(num_cols) + 2), min(12, len(num_cols) + 1)))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                        center=0, ax=ax, linewidths=.5, annot_kws={"size": 8})
            ax.set_title("Correlation Matrix", color="#e2e8f0", fontsize=14)
            plt.tight_layout()
            p = CHART_DIR / "02_correlation_heatmap.png"
            plt.savefig(p, dpi=100, bbox_inches="tight", facecolor="#0d0f1a")
            plt.close()
            charts["Correlation Heatmap"] = str(p)

            # Top correlations
            corr_long = corr.where(~mask).stack().reset_index()
            corr_long.columns = ["feat1", "feat2", "corr"]
            corr_long["abs_corr"] = corr_long["corr"].abs()
            top = corr_long.sort_values("abs_corr", ascending=False).head(5)
            top_corrs = top[["feat1", "feat2", "corr"]].to_dict("records")
            logger.info("Chart saved: correlation heatmap")

        # 3. Boxplots (outlier detection)
        if num_cols:
            fig, axes = plt.subplots(
                max(1, (len(num_cols) + 2) // 3), min(3, len(num_cols)),
                figsize=(15, max(4, 4 * ((len(num_cols) + 2) // 3)))
            )
            axes = np.array(axes).flatten()
            for i, col in enumerate(num_cols[:9]):
                sns.boxplot(y=df[col], ax=axes[i], color="#00d4aa")
                axes[i].set_title(col, color="#e2e8f0")
                q1, q3 = df[col].quantile(.25), df[col].quantile(.75)
                iqr = q3 - q1
                outlier_counts[col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            p = CHART_DIR / "03_boxplots.png"
            plt.savefig(p, dpi=100, bbox_inches="tight", facecolor="#0d0f1a")
            plt.close()
            charts["Boxplots"] = str(p)
            logger.info("Chart saved: boxplots")

        # 4. Categorical bar charts
        for col in cat_cols[:2]:
            vc = df[col].value_counts().head(15)
            value_counts_dict[col] = vc.to_dict()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x=vc.values, y=vc.index, palette="viridis", ax=ax)
            ax.set_title(f"{col} - Value Counts", color="#e2e8f0")
            plt.tight_layout()
            p = CHART_DIR / f"04_barplot_{col}.png"
            plt.savefig(p, dpi=100, bbox_inches="tight", facecolor="#0d0f1a")
            plt.close()
            charts[f"Bar: {col}"] = str(p)
            logger.info(f"Chart saved: bar - {col}")

        # 5. Pairplot (small datasets)
        if 2 <= len(num_cols) <= 5 and len(df) <= 1000:
            pp = sns.pairplot(df[num_cols], diag_kind="kde", plot_kws={"alpha": 0.5, "color": "#6c63ff"})
            pp.fig.suptitle("Pair Plot", color="#e2e8f0", y=1.01)
            p = CHART_DIR / "05_pairplot.png"
            pp.savefig(p, dpi=80, facecolor="#0d0f1a")
            plt.close()
            charts["Pair Plot"] = str(p)
            logger.info("Chart saved: pairplot")

        return {
            "charts":          charts,
            "outlier_counts":  outlier_counts,
            "top_correlations": top_corrs,
            "value_counts":    value_counts_dict,
        }
