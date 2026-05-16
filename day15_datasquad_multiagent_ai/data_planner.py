"""
Agent 1: DataPlanner
Responsibility: Load dataset, profile structure, define analytics strategy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from logger import TeamLogger

logger = TeamLogger("DataPlanner")


class DataPlanner:
    """
    Analyzes raw dataset and outputs a structured plan for downstream agents.
    Determines: feature types, missing data strategy, clustering suitability, target info.
    """

    def run(self, dataset_path: str, target_column: str = None) -> dict:
        logger.info(f"Loading dataset: {dataset_path}")
        df = self._load(dataset_path)

        numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols  = df.select_dtypes(include=["datetime"]).columns.tolist()
        missing_pct    = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        high_missing   = [c for c, v in missing_pct.items() if v > 30]

        plan = {
            "dataset_path":     dataset_path,
            "df":               df,
            "n_rows":           len(df),
            "n_features":       len(df.columns),
            "numeric_cols":     numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols":    datetime_cols,
            "target_column":    target_column,
            "missing_pct":      missing_pct,
            "high_missing_cols": high_missing,
            "drop_cols":        high_missing,
            "clustering_cols":  [c for c in numeric_cols if c != target_column],
            "memory_mb":        round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "has_target":       target_column is not None and target_column in df.columns,
            "strategy": {
                "imputation":    "median" if len(numeric_cols) > 0 else "mode",
                "scaling":       "standard",
                "pca_variance":  0.95,
                "k_range":       (2, 8),
            }
        }

        logger.info(f"Numeric features  : {len(numeric_cols)}")
        logger.info(f"Categorical features: {len(categorical_cols)}")
        logger.info(f"High-missing cols : {high_missing or 'None'}")
        logger.info(f"Clustering on     : {len(plan['clustering_cols'])} features")

        return plan

    def _load(self, path: str) -> pd.DataFrame:
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif ext == ".parquet":
            return pd.read_parquet(path)
        elif ext == ".json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
