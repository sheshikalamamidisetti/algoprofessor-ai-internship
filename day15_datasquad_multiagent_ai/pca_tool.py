"""
PCATool - Agentic Tool for Dimensionality Reduction
Used by MLEngineer agent. Wraps sklearn PCA as a callable tool with structured I/O.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from logger import TeamLogger

logger = TeamLogger("PCATool")
CHART_DIR = Path("outputs/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)


class PCATool:
    """
    Agentic PCA tool. Input: scaled feature matrix + config. Output: structured result dict.
    Automatically selects n_components to retain variance_threshold % of variance.
    """

    name        = "pca_dimensionality_reduction"
    description = "Reduces feature dimensionality while retaining target variance percentage."

    def execute(self, inputs: dict) -> dict:
        X                  = inputs["X"]
        variance_threshold = inputs.get("variance_threshold", 0.95)
        feature_names      = inputs.get("feature_names", [f"f{i}" for i in range(X.shape[1])])

        # Fit PCA to get all components first
        pca_full = PCA()
        pca_full.fit(X)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)

        # Select n_components
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_components = min(n_components, X.shape[1], X.shape[0])

        # Fit final PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_transformed = pca.fit_transform(X)

        explained_pct  = float(np.sum(pca.explained_variance_ratio_) * 100)
        component_vars = pca.explained_variance_ratio_.tolist()
        loadings       = pca.components_.tolist()

        # Chart: Scree plot
        charts = {}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                 facecolor="#0d0f1a")
        # Scree
        axes[0].plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
                     np.cumsum(pca_full.explained_variance_ratio_),
                     marker="o", color="#6c63ff", linewidth=2)
        axes[0].axhline(variance_threshold, color="#00d4aa", linestyle="--", label=f"{variance_threshold*100:.0f}% threshold")
        axes[0].axvline(n_components, color="#ff6584", linestyle="--", label=f"n={n_components}")
        axes[0].set_title("Cumulative Explained Variance", color="#e2e8f0")
        axes[0].set_xlabel("Components", color="#8892a4")
        axes[0].set_ylabel("Cum. Variance", color="#8892a4")
        axes[0].legend(facecolor="#141726", labelcolor="#e2e8f0")
        axes[0].set_facecolor("#141726")
        axes[0].tick_params(colors="#8892a4")

        # Top loadings bar
        if n_components >= 1 and len(feature_names) > 0:
            load_vals = np.abs(pca.components_[0])
            sorted_idx = np.argsort(load_vals)[::-1][:10]
            axes[1].barh([feature_names[i] for i in sorted_idx[::-1]],
                         load_vals[sorted_idx[::-1]], color="#6c63ff")
            axes[1].set_title("PC1 Feature Loadings", color="#e2e8f0")
            axes[1].set_facecolor("#141726")
            axes[1].tick_params(colors="#8892a4")

        plt.tight_layout()
        p = CHART_DIR / "06_pca_scree.png"
        plt.savefig(p, dpi=100, bbox_inches="tight", facecolor="#0d0f1a")
        plt.close()
        charts["PCA Scree Plot"] = str(p)

        logger.info(f"PCA: {n_components} components, {explained_pct:.1f}% variance explained")

        return {
            "n_components":          n_components,
            "explained_variance_pct": explained_pct,
            "component_variances":   component_vars,
            "loadings":              loadings,
            "X_transformed":         X_transformed,
            "feature_names":         feature_names,
            "charts":                charts,
        }
