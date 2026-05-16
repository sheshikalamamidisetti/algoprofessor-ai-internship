"""
Agent 3: MLEngineer
Responsibility: Transform ML algorithms into agentic tools.
  - > PCATool       : dimensionality reduction pipeline
  - > KMeansTool    : clustering orchestrator with auto-K selection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from logger import TeamLogger

logger = TeamLogger("MLEngineer")


class MLEngineer:
    """
    Orchestrates PCA and KMeans as agentic tools.
    Decides when to apply each, passes results between tools, interprets output.
    """

    def __init__(self, pca_tool, clustering_tool):
        self.pca_tool         = pca_tool
        self.clustering_tool  = clustering_tool

    def run(self, plan: dict, eda_results: dict) -> dict:
        df_clean  = eda_results["df_clean"]
        clust_cols = [c for c in plan["clustering_cols"] if c in df_clean.columns]

        if len(clust_cols) < 2:
            logger.warning("Not enough numeric features for clustering. Skipping.")
            return {"skipped": True}

        # Scale features
        logger.info(f"Scaling {len(clust_cols)} features for ML pipeline...")
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(df_clean[clust_cols])

        # ── AGENTIC TOOL 1: PCA ──────────────────────────────────────────────
        logger.info("Invoking PCATool (dimensionality reduction)...")
        pca_result = self.pca_tool.execute({
            "X": X_scaled,
            "variance_threshold": plan["strategy"]["pca_variance"],
            "feature_names": clust_cols,
        })
        logger.info(f"PCA: {pca_result['n_components']} components retain "
                    f"{pca_result['explained_variance_pct']:.1f}% variance")

        # ── AGENTIC TOOL 2: KMeans ───────────────────────────────────────────
        logger.info("Invoking KMeansClusteringTool (auto-K orchestration)...")
        k_min, k_max = plan["strategy"]["k_range"]
        kmeans_result = self.clustering_tool.execute({
            "X_pca":   pca_result["X_transformed"],
            "X_orig":  X_scaled,
            "k_range": (k_min, k_max),
            "feature_names": clust_cols,
        })
        logger.info(f"KMeans: optimal K={kmeans_result['optimal_k']}, "
                    f"silhouette={kmeans_result['silhouette_score']:.3f}")

        # Attach cluster labels back to clean df
        df_clean = df_clean.copy()
        df_clean["cluster"] = kmeans_result["labels"]

        cluster_profiles = (
            df_clean.groupby("cluster")[clust_cols].mean().round(3).to_dict()
        )

        return {
            "clust_cols":           clust_cols,
            "X_scaled":             X_scaled,
            "pca":                  pca_result,
            "optimal_k":            kmeans_result["optimal_k"],
            "labels":               kmeans_result["labels"],
            "silhouette_score":     kmeans_result["silhouette_score"],
            "inertia_curve":        kmeans_result["inertia_curve"],
            "silhouette_scores_all": kmeans_result["silhouette_scores_all"],
            "cluster_sizes":        kmeans_result["cluster_sizes"],
            "cluster_profiles":     cluster_profiles,
            "df_labeled":           df_clean,
            "charts":               {**pca_result.get("charts", {}), **kmeans_result.get("charts", {})},
        }
