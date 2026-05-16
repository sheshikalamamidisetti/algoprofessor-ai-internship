"""
KMeansClusteringTool - Agentic Tool for Clustering Orchestration
Used by MLEngineer agent.
Auto-selects optimal K using Elbow + Silhouette methods.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pathlib import Path

from logger import TeamLogger

logger = TeamLogger("KMeansTool")

CHART_DIR = Path("outputs/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)


class KMeansClusteringTool:
    """
    Agentic KMeans orchestrator.
    Runs K from k_min to k_max,
    picks best K,
    returns labels + diagnostics.
    """

    name = "kmeans_clustering_orchestrator"

    description = (
        "Auto-selects optimal K using elbow + silhouette methods."
    )

    def execute(self, inputs: dict) -> dict:

        X_pca = inputs["X_pca"]

        X_orig = inputs.get("X_orig", X_pca)

        k_min, k_max = inputs.get("k_range", (2, 8))

        feature_names = inputs.get("feature_names", [])

        k_max = min(k_max, len(X_pca) - 1)

        inertias = []
        silhouette_scores = []
        models = {}

        logger.info(f"Testing K = {k_min} to {k_max}...")

        # =====================================================
        # TEST MULTIPLE K VALUES
        # =====================================================
        for k in range(k_min, k_max + 1):

            km = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
                max_iter=300
            )

            labels = km.fit_predict(X_pca)

            inertias.append(km.inertia_)

            sil = (
                silhouette_score(X_pca, labels)
                if len(set(labels)) > 1
                else 0.0
            )

            silhouette_scores.append(sil)

            models[k] = (km, labels, sil)

            # FIXED WINDOWS-SAFE LOGGING
            logger.info(
                f"K={k} -> inertia={km.inertia_:.1f}, silhouette={sil:.4f}"
            )

        # =====================================================
        # BEST K
        # =====================================================
        best_idx = int(np.argmax(silhouette_scores))

        optimal_k = k_min + best_idx

        best_km, best_labels, best_sil = models[optimal_k]

        cluster_sizes = [
            int((best_labels == k).sum())
            for k in range(optimal_k)
        ]

        # =====================================================
        # CHARTS
        # =====================================================
        charts = {}

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 4),
            facecolor="#0d0f1a"
        )

        ks = list(range(k_min, k_max + 1))

        # =====================================================
        # ELBOW CURVE
        # =====================================================
        axes[0].plot(
            ks,
            inertias,
            marker="o",
            color="#6c63ff",
            linewidth=2
        )

        axes[0].axvline(
            optimal_k,
            color="#ff6584",
            linestyle="--",
            label=f"K={optimal_k}"
        )

        axes[0].set_title(
            "Elbow Curve",
            color="#e2e8f0"
        )

        axes[0].set_xlabel(
            "K",
            color="#8892a4"
        )

        axes[0].set_ylabel(
            "Inertia",
            color="#8892a4"
        )

        axes[0].legend(
            facecolor="#141726",
            labelcolor="#e2e8f0"
        )

        axes[0].set_facecolor("#141726")

        axes[0].tick_params(colors="#8892a4")

        # =====================================================
        # SILHOUETTE CHART
        # =====================================================
        axes[1].bar(
            ks,
            silhouette_scores,
            color=[
                "#00d4aa" if k == optimal_k else "#6c63ff"
                for k in ks
            ]
        )

        axes[1].set_title(
            "Silhouette Scores",
            color="#e2e8f0"
        )

        axes[1].set_xlabel(
            "K",
            color="#8892a4"
        )

        axes[1].set_ylabel(
            "Score",
            color="#8892a4"
        )

        axes[1].set_facecolor("#141726")

        axes[1].tick_params(colors="#8892a4")

        plt.tight_layout()

        p = CHART_DIR / "07_kmeans_diagnostics.png"

        plt.savefig(
            p,
            dpi=100,
            bbox_inches="tight",
            facecolor="#0d0f1a"
        )

        plt.close()

        charts["KMeans Diagnostics"] = str(p)

        # =====================================================
        # PCA CLUSTER SCATTER
        # =====================================================
        if X_pca.shape[1] >= 2:

            fig, ax = plt.subplots(
                figsize=(9, 6),
                facecolor="#0d0f1a"
            )

            scatter = ax.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                c=best_labels,
                cmap="tab10",
                alpha=0.7,
                s=20
            )

            centers = best_km.cluster_centers_

            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                c="red",
                marker="X",
                s=150,
                label="Centroids",
                zorder=5
            )

            ax.set_title(
                f"K={optimal_k} Clusters (PCA space)",
                color="#e2e8f0",
                fontsize=13
            )

            ax.set_xlabel(
                "PC1",
                color="#8892a4"
            )

            ax.set_ylabel(
                "PC2",
                color="#8892a4"
            )

            ax.legend(
                facecolor="#141726",
                labelcolor="#e2e8f0"
            )

            ax.set_facecolor("#141726")

            ax.tick_params(colors="#8892a4")

            plt.colorbar(
                scatter,
                ax=ax,
                label="Cluster"
            )

            plt.tight_layout()

            p2 = CHART_DIR / "08_cluster_scatter.png"

            plt.savefig(
                p2,
                dpi=100,
                bbox_inches="tight",
                facecolor="#0d0f1a"
            )

            plt.close()

            charts["Cluster Scatter (PCA)"] = str(p2)

        # =====================================================
        # RETURN RESULTS
        # =====================================================
        return {

            "optimal_k": optimal_k,

            "labels": best_labels,

            "silhouette_score": best_sil,

            "inertia_curve": list(zip(ks, inertias)),

            "silhouette_scores_all": list(
                zip(ks, silhouette_scores)
            ),

            "cluster_sizes": cluster_sizes,

            "model": best_km,

            "charts": charts,
        }