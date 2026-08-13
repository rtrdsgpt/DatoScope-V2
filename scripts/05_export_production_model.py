"""
05_export_production_model.py
DatoScope — Step 5: Export a Production-stage MLflow model for DVC tracking

Pulls the current Production version of a registered model (see
api/tracking.py — registered as "<dataset_name>__<task>") out of the MLflow
Model Registry and saves it to models/<registered_name>.pkl in the same
{"model", "features", "target", "task_type"} shape utils.modeling.
export_model_bytes/the API's download endpoint already use, so a
DVC-tracked model file is loadable the same way as a freshly downloaded one.

This only writes the file — versioning it is a deliberate, reviewable git
step, not something this script does on its own:

    python scripts/05_export_production_model.py <dataset_name>__<task>
    dvc add models/<dataset_name>__<task>.pkl
    git add models/<dataset_name>__<task>.pkl.dvc models/.gitignore
    git commit -m "Track <dataset_name>__<task> Production model"
    dvc push
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import mlflow
from mlflow.tracking import MlflowClient

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def export(registered_name: str) -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("S3_ACCESS_KEY", "datoscope"))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("S3_SECRET_KEY", "datoscope123"))

    client = MlflowClient()
    versions = client.get_latest_versions(registered_name, stages=["Production"])
    if not versions:
        print(f"No Production version found for '{registered_name}'. Promote one first:", file=sys.stderr)
        print(f"  POST /models/{registered_name}/promote", file=sys.stderr)
        raise SystemExit(1)
    version = versions[0]

    model = mlflow.sklearn.load_model(f"models:/{registered_name}/Production")
    run = client.get_run(version.run_id)
    params = run.data.params
    dataset_name, _, task = registered_name.partition("__")

    payload = {
        "model": model,
        "features": params.get("features", "").split(",") if params.get("features") else [],
        "target": params.get("target_col", ""),
        "task_type": task,
        "dataset_name": dataset_name,
        "mlflow_run_id": version.run_id,
        "mlflow_version": version.version,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"{registered_name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Exported '{registered_name}' v{version.version} (run {version.run_id}) -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registered_name", help="e.g. iris__classification")
    args = parser.parse_args()
    export(args.registered_name)
