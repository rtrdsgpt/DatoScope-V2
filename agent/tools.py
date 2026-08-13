"""
Tool functions the autonomous pipeline agent (agent/pipeline_agent.py) and
the MCP server (agent/mcp_server.py) both call — one implementation, two
callers, per todo.md section 7's MCP requirement ("drive DatoScope's
analysis tools directly").

Each wraps utils/api_client.py's already-built, already-tested HTTP calls
to the DatoScope API rather than re-implementing them. Training tools trim
the response to just the metric fields utils.comparison actually reads
(dropping y_test/y_pred/confusion matrices/coefficients) — small enough for
an LLM tool-call round trip, and the trimmed dict is still sufficient input
for compare_models, so results flow forward through the tool-call chain
without the agent needing to cache anything server-side.
"""

from __future__ import annotations

from utils import api_client
from utils.comparison import score_classification_models, score_clustering_models, score_regression_models

_REGRESSION_METRIC_KEYS = ["R2", "CV_R2", "RMSE", "MAE", "Overfit_Gap", "mlflow_run_id", "model_id"]
_CLASSIFICATION_METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1", "Macro_F1", "CV_Accuracy", "mlflow_run_id", "model_id"]
_CLUSTERING_METRIC_KEYS = ["n_clusters", "Silhouette", "Davies_Bouldin", "Calinski_Harabasz", "FM_Score", "Rand_Index", "mlflow_run_id"]


def _trim(results: dict, keys: list[str]) -> dict:
    return {name: {k: row[k] for k in keys if k in row} for name, row in results.items()}


def generate_dataset(
    dataset_name: str,
    task_type: str,
    dataset_type: str,
    n_samples: int = 500,
    n_features: int = 6,
    target_name: str = "target",
    noise: float = 0.15,
    random_seed: int = 42,
) -> dict:
    """Generate a synthetic dataset and land it in the raw zone. task_type is one of
    Regression/Classification/Clustering. dataset_type depends on task_type — Regression:
    Linear/Nonlinear/Sinusoidal/High-Dimensional; Classification: Linearly Separable/Overlapping
    Classes/Imbalanced Classes; Clustering: Gaussian Blobs/Two Moons/Concentric Circles/Spiral/
    Anisotropic Blobs/Variable Density Blobs."""
    return api_client.generate_dataset(
        dataset_name=dataset_name,
        task_type=task_type,
        dataset_type=dataset_type,
        n_samples=n_samples,
        n_features=n_features,
        target_name=target_name,
        noise=noise,
        random_seed=random_seed,
    )


def kaggle_dataset(handle: str, dataset_name: str | None = None) -> dict:
    """Pull a real dataset from Kaggle (e.g. handle="uciml/iris") and land it in the raw zone."""
    return api_client.kaggle_dataset(handle, dataset_name)


def clean_dataset(
    dataset_name: str,
    run_id: str | None = None,
    missing_strategy: str = "mean",
    outlier_method: str = "IQR",
    scale_method: str = "Standard",
    remove_dupes: bool = True,
    label_col: str | None = None,
) -> dict:
    """Transform + validate + load a raw extract into the warehouse. run_id is OPTIONAL: omit it
    entirely to use the latest raw extract for dataset_name (do not invent a placeholder value —
    only pass an actual run_id string a previous tool call returned). missing_strategy: mean/
    median/mode/drop. outlier_method: IQR/Z-Score/none. scale_method: Standard/MinMax/Robust."""
    return api_client.clean_dataset(
        dataset_name,
        run_id=run_id,
        missing_strategy=missing_strategy,
        outlier_method=outlier_method,
        scale_method=scale_method,
        remove_dupes=remove_dupes,
        label_col=label_col,
    )


def eda_summary(dataset_name: str, run_id: str | None = None) -> dict:
    """Get summary statistics (count/mean/std/min/max/skewness/kurtosis per numeric column) for
    a cleaned dataset already in the warehouse."""
    return api_client.eda_summary(dataset_name, run_id=run_id)


def train_regression(
    dataset_name: str,
    features: list[str],
    target_col: str,
    run_id: str | None = None,
    run_lr: bool = True,
    run_ridge: bool = True,
    run_lasso: bool = True,
) -> dict:
    """Train regression models (Linear/Ridge/Lasso) on a cleaned dataset. Returns per-model
    R2/CV_R2/RMSE/MAE/Overfit_Gap plus a model_id/mlflow_run_id for each."""
    results = api_client.train_regression(
        dataset_name, run_id=run_id, features=features, target_col=target_col,
        run_lr=run_lr, run_ridge=run_ridge, run_lasso=run_lasso,
    )
    return _trim(results, _REGRESSION_METRIC_KEYS)


def train_classification(
    dataset_name: str,
    features: list[str],
    target_col: str,
    run_id: str | None = None,
    run_logreg: bool = True,
    run_rf: bool = True,
    run_knn: bool = True,
) -> dict:
    """Train classification models (Logistic Regression/Random Forest/KNN) on a cleaned dataset.
    Returns per-model Accuracy/Precision/Recall/F1/Macro_F1/CV_Accuracy plus a model_id/
    mlflow_run_id for each."""
    results = api_client.train_classification(
        dataset_name, run_id=run_id, features=features, target_col=target_col,
        run_logreg=run_logreg, run_rf=run_rf, run_knn=run_knn,
    )
    return _trim(results, _CLASSIFICATION_METRIC_KEYS)


def train_clustering(
    dataset_name: str,
    features: list[str],
    run_id: str | None = None,
    run_km: bool = True,
    run_db: bool = True,
    run_hc: bool = True,
    k_val: int = 3,
    ground_truth_col: str | None = None,
) -> dict:
    """Train clustering models (K-Means/DBSCAN/Hierarchical) on a cleaned dataset. Returns
    per-algorithm n_clusters/Silhouette/Davies_Bouldin/Calinski_Harabasz (+ FM_Score/Rand_Index if
    ground_truth_col is given) plus an mlflow_run_id for each."""
    result = api_client.run_clustering(
        dataset_name, run_id=run_id, features=features,
        run_km=run_km, run_db=run_db, run_hc=run_hc, k_val=k_val, ground_truth_col=ground_truth_col,
    )
    return _trim(result["models"], _CLUSTERING_METRIC_KEYS)


def compare_models(task: str, results: dict, dataset_name: str | None = None) -> dict:
    """Pick the winning model from a set of training results (as returned by train_regression/
    train_classification/train_clustering). task is one of regression/classification/clustering.
    If dataset_name is given, the winner is registered into the MLflow Model Registry (Staging)."""
    if not isinstance(results, dict) or not results or not all(isinstance(v, dict) for v in results.values()):
        raise ValueError(
            "compare_models needs the exact dict a train_regression/train_classification/train_clustering "
            "call returned (model name -> metrics dict). Got something else — if the training call failed, "
            "fix that first; do not call compare_models with an error result or invented data."
        )
    scorer = {"regression": score_regression_models, "classification": score_classification_models, "clustering": score_clustering_models}[task]
    scored = scorer(results)
    if dataset_name:
        try:
            api_result = {"regression": api_client.compare_regression, "classification": api_client.compare_classification, "clustering": api_client.compare_clustering}[task](results, dataset_name=dataset_name)
            scored["mlflow_registration"] = api_result.get("mlflow_registration")
            if api_result.get("mlflow_error"):
                scored["mlflow_error"] = api_result["mlflow_error"]
        except api_client.ApiError as exc:
            # Surfaced, not swallowed: a caller (LLM or human) reading this
            # result needs to know registration didn't happen and why,
            # rather than a bare `mlflow_registration: null` that looks
            # identical to "registration wasn't attempted."
            scored["mlflow_registration"] = None
            scored["mlflow_error"] = f"MLflow registration request failed: {exc}"
    return scored


TOOLS = [generate_dataset, kaggle_dataset, clean_dataset, eda_summary, train_regression, train_classification, train_clustering, compare_models]
