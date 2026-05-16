"""
M10: DataSquad - 4-Agent Analytics Team
Orchestrates:
DataPlanner -> StatAnalyst -> MLEngineer -> ReportWriter
"""

import asyncio
import time
from pathlib import Path

# Direct imports (all files are in same folder)
from data_planner import DataPlanner
from stat_analyst import StatAnalyst
from ml_engineer import MLEngineer
from report_writer import ReportWriter
from eda_pipeline import AutoEDAPipeline
from clustering_tool import KMeansClusteringTool
from pca_tool import PCATool
from logger import TeamLogger

logger = TeamLogger("DataSquad")


async def run_datasquad(
    dataset_path: str,
    target_column: str = None,
    output_dir: str = "outputs"
):

    # Create outputs folder
    Path(output_dir).mkdir(exist_ok=True)

    start = time.time()

    logger.header("DataSquad Analytics Team - Starting Mission")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output : {output_dir}/")

    # =====================================================
    # AGENT 1 - DataPlanner
    # =====================================================
    logger.agent(
        "DataPlanner",
        "Analyzing dataset and defining analytics strategy..."
    )

    planner = DataPlanner()

    plan = planner.run(
        dataset_path,
        target_column
    )

    logger.success(
        "DataPlanner",
        f"Strategy ready - {plan['n_features']} features, {plan['n_rows']} rows"
    )

    # =====================================================
    # AGENT 2 - StatAnalyst
    # =====================================================
    logger.agent(
        "StatAnalyst",
        "Running automated EDA pipeline..."
    )

    analyst = StatAnalyst(
        eda_pipeline=AutoEDAPipeline()
    )

    eda_results = analyst.run(plan)

    logger.success(
        "StatAnalyst",
        f"EDA complete - {len(eda_results['charts'])} charts generated"
    )

    # =====================================================
    # AGENT 3 - MLEngineer
    # =====================================================
    logger.agent(
        "MLEngineer",
        "Running PCA + KMeans clustering pipeline..."
    )

    ml_engineer = MLEngineer(
        pca_tool=PCATool(),
        clustering_tool=KMeansClusteringTool()
    )

    ml_results = ml_engineer.run(
        plan,
        eda_results
    )

    logger.success(
        "MLEngineer",
        f"Clustering complete - {ml_results['optimal_k']} clusters found"
    )

    # =====================================================
    # AGENT 4 - ReportWriter
    # =====================================================
    logger.agent(
        "ReportWriter",
        "Compiling analytics report..."
    )

    writer = ReportWriter(
        output_dir=output_dir
    )

    report_path = writer.run(
        plan,
        eda_results,
        ml_results
    )

    logger.success(
        "ReportWriter",
        f"Report saved -> {report_path}"
    )

    # =====================================================
    # FINISH
    # =====================================================
    elapsed = round(time.time() - start, 2)

    logger.header(
        f"Mission Complete in {elapsed}s - Report: {report_path}"
    )

    return {
        "plan": plan,
        "eda": eda_results,
        "ml": ml_results,
        "report": report_path,
        "elapsed_seconds": elapsed
    }


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":

    import sys

    dataset = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "sample_customers.csv"
    )

    target = (
        sys.argv[2]
        if len(sys.argv) > 2
        else None
    )

    asyncio.run(
        run_datasquad(dataset, target)
    )