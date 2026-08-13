"""
JSON-safe conversion for utils.modeling's results, which contain trained
sklearn model objects, numpy arrays, and pandas Series — none of them
directly JSON-serializable. Also renames metric keys to the plain
identifiers utils.comparison expects (no unicode/spaces), so a modeling
endpoint's response can be fed straight into a comparison endpoint.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

# In-memory, single-process model registry for the download endpoint —
# acceptable for a local dev API (mirrors today's session-scoped download
# button; not meant to survive a restart or scale across workers).
MODEL_REGISTRY: dict[str, dict] = {}


def _to_list(value):
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def serialize_regression_results(results: dict, *, dataset_name: str, target_col: str) -> dict:
    out = {}
    for name, row in results.items():
        model_id = str(uuid.uuid4())
        MODEL_REGISTRY[model_id] = {
            "model": row["model"],
            "features": row["features"],
            "target": target_col,
            "task_type": "Regression",
        }
        out[name] = {
            "R2": row["R²"],
            "MSE": row["MSE"],
            "RMSE": row["RMSE"],
            "MAE": row["MAE"],
            "CV_R2": row["CV R²"],
            "Overfit_Gap": row["Overfit Gap"],
            "split_method": row["split_method"],
            "features": row["features"],
            "target": row["target"],
            "coef": row.get("coef"),
            "y_test": _to_list(row["y_test"]),
            "y_pred": _to_list(row["y_pred"]),
            "model_id": model_id,
        }
    return out


def serialize_classification_results(results: dict, *, dataset_name: str, target_col: str) -> dict:
    out = {}
    for name, row in results.items():
        model_id = str(uuid.uuid4())
        MODEL_REGISTRY[model_id] = {
            "model": row["model"],
            "features": row["features"],
            "target": target_col,
            "task_type": "Classification",
        }
        out[name] = {
            "Accuracy": row["Accuracy"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
            "F1": row["F1"],
            "Macro_F1": row["Macro F1"],
            "CV_Accuracy": row["CV Accuracy"],
            "confusion_matrix": _to_list(row["Confusion Matrix"]),
            "report": row["Report"],
            "split_method": row["split_method"],
            "features": row["features"],
            "target": row["target"],
            "y_test": _to_list(row["y_test"]),
            "y_pred": _to_list(row["y_pred"]),
            "model_id": model_id,
        }
    return out


def serialize_clustering_results(result: dict) -> dict:
    models = {}
    for name, row in result["results"].items():
        models[name] = {
            "n_clusters": row["n_clusters"],
            "labels": _to_list(row["labels"]),
            "Silhouette": row.get("Silhouette"),
            "Davies_Bouldin": row.get("Davies-Bouldin"),
            "Calinski_Harabasz": row.get("Calinski-Harabasz"),
            "FM_Score": row.get("FM Score"),
            "Rand_Index": row.get("Rand Index"),
        }
    # Scaled feature matrix — needed client-side for the dendrogram/elbow-curve
    # diagnostics, which operate directly on the matrix rather than on
    # already-computed metrics (see pages/3_Clustering.py).
    return {"models": models, "features": result["features"], "X": result["X"].to_dict(orient="records")}
