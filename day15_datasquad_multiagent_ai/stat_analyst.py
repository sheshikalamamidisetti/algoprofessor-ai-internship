"""
Agent 2: StatAnalyst
Responsibility: Run automated EDA pipeline. Produces statistical summaries + charts.
Extends: NumPy / Pandas / Seaborn stack via AutoEDAPipeline.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from logger import TeamLogger

logger = TeamLogger("StatAnalyst")


class StatAnalyst:
    """
    Consumes a DataPlanner plan, runs the EDA pipeline, and returns structured results.
    Charts are saved to outputs/charts/.
    """

    def __init__(self, eda_pipeline):
        self.eda = eda_pipeline

    def run(self, plan: dict) -> dict:
        df      = plan["df"]
        num_cols = plan["numeric_cols"]
        cat_cols = plan["categorical_cols"]

        # Clean for EDA
        df_clean = self._preprocess(df, plan)

        # Run automated EDA
        eda_report = self.eda.run(df_clean, num_cols, cat_cols)

        # Statistical summaries
        desc_stats = df_clean[num_cols].describe().round(4).to_dict() if num_cols else {}
        skewness   = df_clean[num_cols].skew().round(4).to_dict() if num_cols else {}
        kurtosis   = df_clean[num_cols].kurt().round(4).to_dict() if num_cols else {}

        # Correlation matrix
        corr_matrix = None
        if len(num_cols) > 1:
            corr_matrix = df_clean[num_cols].corr().round(4).to_dict()

        results = {
            "df_clean":      df_clean,
            "desc_stats":    desc_stats,
            "skewness":      skewness,
            "kurtosis":      kurtosis,
            "corr_matrix":   corr_matrix,
            "charts":        eda_report["charts"],
            "outlier_counts": eda_report["outlier_counts"],
            "top_correlations": eda_report.get("top_correlations", []),
            "value_counts":  eda_report.get("value_counts", {}),
        }

        logger.info(f"Descriptive stats computed for {len(num_cols)} numeric columns")
        logger.info(f"Outliers flagged in: {[k for k,v in results['outlier_counts'].items() if v > 0]}")

        return results

    def _preprocess(self, df: pd.DataFrame, plan: dict) -> pd.DataFrame:
        df = df.drop(columns=plan["drop_cols"], errors="ignore").copy()
        strategy = plan["strategy"]["imputation"]

        for col in plan["numeric_cols"]:
            if col in df.columns and df[col].isnull().any():
                fill = df[col].median() if strategy == "median" else df[col].mean()
                df[col] = df[col].fillna(fill)

        for col in plan["categorical_cols"]:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        return df
