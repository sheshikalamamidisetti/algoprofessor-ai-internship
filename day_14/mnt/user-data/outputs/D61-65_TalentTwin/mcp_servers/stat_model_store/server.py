"""
MCP Server: Statistical Model Store
Endpoints: PCA (dimensionality reduction), SVM (classification), RF (regression/classification)
Run: python mcp_servers/stat_model_store/server.py
"""
from __future__ import annotations

import asyncio
import json
import pickle
import os
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

app = Server("stat-model-store")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Helper: serialise numpy arrays ───────────────────────────────────────────

def np_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: np_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [np_to_list(i) for i in obj]
    return obj


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # PCA
        Tool(
            name="pca_fit_transform",
            description="Fit PCA on data and return reduced representation + explained variance",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "array", "description": "2D array (list of rows)"},
                    "n_components": {"type": "integer", "default": 2},
                    "model_name": {"type": "string", "description": "Save model with this name"}
                },
                "required": ["data"]
            }
        ),
        Tool(
            name="pca_transform",
            description="Transform new data using a previously fitted PCA model",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "array"},
                    "model_name": {"type": "string"}
                },
                "required": ["data", "model_name"]
            }
        ),
        # SVM
        Tool(
            name="svm_train",
            description="Train an SVM classifier or regressor",
            inputSchema={
                "type": "object",
                "properties": {
                    "X": {"type": "array", "description": "Feature matrix"},
                    "y": {"type": "array", "description": "Labels"},
                    "task": {"type": "string", "enum": ["classify", "regress"], "default": "classify"},
                    "kernel": {"type": "string", "enum": ["rbf", "linear", "poly"], "default": "rbf"},
                    "model_name": {"type": "string"}
                },
                "required": ["X", "y", "model_name"]
            }
        ),
        Tool(
            name="svm_predict",
            description="Predict using a saved SVM model",
            inputSchema={
                "type": "object",
                "properties": {
                    "X": {"type": "array"},
                    "model_name": {"type": "string"}
                },
                "required": ["X", "model_name"]
            }
        ),
        # Random Forest
        Tool(
            name="rf_train",
            description="Train a Random Forest model for classification or regression",
            inputSchema={
                "type": "object",
                "properties": {
                    "X": {"type": "array"},
                    "y": {"type": "array"},
                    "task": {"type": "string", "enum": ["classify", "regress"], "default": "classify"},
                    "n_estimators": {"type": "integer", "default": 100},
                    "model_name": {"type": "string"}
                },
                "required": ["X", "y", "model_name"]
            }
        ),
        Tool(
            name="rf_predict",
            description="Predict and get feature importances from a saved RF model",
            inputSchema={
                "type": "object",
                "properties": {
                    "X": {"type": "array"},
                    "model_name": {"type": "string"},
                    "return_importances": {"type": "boolean", "default": False}
                },
                "required": ["X", "model_name"]
            }
        ),
        Tool(
            name="list_models",
            description="List all saved models in the model store",
            inputSchema={"type": "object", "properties": {}}
        )
    ]


def model_path(name: str) -> str:
    return os.path.join(MODEL_DIR, f"{name}.joblib")


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "pca_fit_transform":
            X = np.array(arguments["data"])
            n = arguments.get("n_components", 2)
            pca = PCA(n_components=min(n, X.shape[1]))
            reduced = pca.fit_transform(X)
            mname = arguments.get("model_name", "pca_default")
            joblib.dump(pca, model_path(mname))
            result = {
                "reduced_shape": list(reduced.shape),
                "reduced_data": reduced.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "total_variance_explained": float(pca.explained_variance_ratio_.sum()),
                "model_saved": mname
            }

        elif name == "pca_transform":
            pca = joblib.load(model_path(arguments["model_name"]))
            reduced = pca.transform(np.array(arguments["data"]))
            result = {"reduced_data": reduced.tolist()}

        elif name == "svm_train":
            X, y = np.array(arguments["X"]), np.array(arguments["y"])
            task = arguments.get("task", "classify")
            kernel = arguments.get("kernel", "rbf")
            if task == "classify":
                model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel=kernel, probability=True))])
            else:
                model = Pipeline([("scaler", StandardScaler()), ("svm", SVR(kernel=kernel))])
            model.fit(X, y)
            joblib.dump({"model": model, "task": task}, model_path(arguments["model_name"]))
            score = model.score(X, y)
            result = {"trained": True, "train_score": round(score, 4), "model_saved": arguments["model_name"]}

        elif name == "svm_predict":
            saved = joblib.load(model_path(arguments["model_name"]))
            model, task = saved["model"], saved["task"]
            X = np.array(arguments["X"])
            preds = model.predict(X)
            result = {"predictions": preds.tolist()}
            if task == "classify" and hasattr(model.named_steps["svm"], "predict_proba"):
                proba = model.predict_proba(X)
                result["probabilities"] = proba.tolist()

        elif name == "rf_train":
            X, y = np.array(arguments["X"]), np.array(arguments["y"])
            task = arguments.get("task", "classify")
            n_est = arguments.get("n_estimators", 100)
            if task == "classify":
                model = RandomForestClassifier(n_estimators=n_est, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=n_est, random_state=42)
            model.fit(X, y)
            joblib.dump({"model": model, "task": task}, model_path(arguments["model_name"]))
            result = {
                "trained": True,
                "oob_available": False,
                "n_features": model.n_features_in_,
                "model_saved": arguments["model_name"]
            }

        elif name == "rf_predict":
            saved = joblib.load(model_path(arguments["model_name"]))
            model = saved["model"]
            X = np.array(arguments["X"])
            preds = model.predict(X)
            result = {"predictions": preds.tolist()}
            if arguments.get("return_importances", False):
                result["feature_importances"] = model.feature_importances_.tolist()

        elif name == "list_models":
            files = os.listdir(MODEL_DIR)
            result = {"models": [f.replace(".joblib", "") for f in files if f.endswith(".joblib")]}

        else:
            result = {"error": f"Unknown tool: {name}"}

    except Exception as e:
        result = {"error": str(e)}

    return [TextContent(type="text", text=json.dumps(np_to_list(result)))]


async def main():
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
