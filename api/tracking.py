"""
MLflow integration — logs every trained model as an MLflow run (params +
metrics + the model artifact), and promotes a comparison's winning model
into the MLflow Model Registry (Staging, then Production on request).

Requires MLFLOW_TRACKING_URI (defaults to the docker-compose service) and,
for artifact upload, the same MinIO credentials the ETL pipeline uses
(MLFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) —
MLflow's S3 artifact store is written to directly by the client, not
proxied through the tracking server.
"""

from __future__ import annotations

import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))
os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("S3_ACCESS_KEY", "datoscope"))
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("S3_SECRET_KEY", "datoscope123"))


class ModelPromotionError(Exception):
    pass


def _registered_name(dataset_name: str, task: str) -> str:
    return f"{dataset_name}__{task}"


def log_model_run(*, dataset_name: str, task: str, model_name: str, model, params: dict, metrics: dict) -> str:
    """Log one trained model as its own MLflow run. Returns the run_id."""
    mlflow.set_experiment(f"{dataset_name}__{task}")
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        # skops (mlflow's default sklearn serialization) rejects some models'
        # internal types (e.g. KNeighborsClassifier's KD-tree) unless each is
        # explicitly allow-listed; pickle handles every sklearn estimator here
        # uniformly and matches what utils.modeling.export_model_bytes already
        # uses for the app's own model-download feature.
        mlflow.sklearn.log_model(model, name="model", serialization_format="pickle")
    return run.info.run_id


def register_and_stage(*, dataset_name: str, task: str, run_id: str, stage: str = "Staging") -> dict:
    """Register the model from `run_id` and transition the new version to `stage`."""
    registered_name = _registered_name(dataset_name, task)
    result = mlflow.register_model(f"runs:/{run_id}/model", registered_name)
    client = MlflowClient()
    client.transition_model_version_stage(name=registered_name, version=result.version, stage=stage)
    return {"registered_name": registered_name, "version": result.version, "stage": stage}


def promote_to_production(registered_name: str) -> dict:
    """Move the latest Staging version of a registered model to Production."""
    client = MlflowClient()
    try:
        staging = client.get_latest_versions(registered_name, stages=["Staging"])
    except MlflowException as exc:
        raise ModelPromotionError(f"No registered model '{registered_name}'") from exc
    if not staging:
        raise ModelPromotionError(f"No Staging version found for '{registered_name}'")
    latest = staging[0]
    client.transition_model_version_stage(
        name=registered_name, version=latest.version, stage="Production", archive_existing_versions=True
    )
    return {"registered_name": registered_name, "version": latest.version, "stage": "Production"}


def get_model_versions(registered_name: str) -> list[dict]:
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{registered_name}'")
    except Exception:
        return []
    return [{"version": v.version, "stage": v.current_stage, "run_id": v.run_id} for v in versions]
