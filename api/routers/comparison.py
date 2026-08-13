"""
Comparison endpoints — stateless winner-selection over metrics a client
already has (e.g. straight from a /modeling or /clustering response).
Mirrors pages/4_Comparison.py's scoring logic (utils/comparison.py).

When `dataset_name` is given, the winning model (identified by the
mlflow_run_id the modeling/clustering endpoint already attached to it) is
registered into the MLflow Model Registry and moved to Staging — "log every
model-comparison run ... promote the best model per dataset-type" from
todo.md section 4. Best-effort: comparison itself still succeeds if MLflow
registration fails (e.g. the run has no logged model artifact).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import ComparisonRequest
from api.tracking import register_and_stage
from utils.comparison import score_classification_models, score_clustering_models, score_regression_models

router = APIRouter(prefix="/comparison", tags=["comparison"])


def _maybe_register_winner(scored: dict, req: ComparisonRequest, task: str) -> dict:
    winner = scored.get("winner")
    if not winner or not req.dataset_name:
        return scored
    run_id = req.results.get(winner, {}).get("mlflow_run_id")
    if not run_id:
        return scored
    try:
        scored["mlflow_registration"] = register_and_stage(
            dataset_name=req.dataset_name, task=task, run_id=run_id, stage="Staging"
        )
    except Exception as exc:
        scored["mlflow_registration"] = None
        scored["mlflow_error"] = str(exc)
    return scored


@router.post("/regression")
def regression(req: ComparisonRequest) -> dict:
    if not req.results:
        raise HTTPException(status_code=422, detail="results is empty")
    return _maybe_register_winner(score_regression_models(req.results), req, "regression")


@router.post("/classification")
def classification(req: ComparisonRequest) -> dict:
    if not req.results:
        raise HTTPException(status_code=422, detail="results is empty")
    return _maybe_register_winner(score_classification_models(req.results), req, "classification")


@router.post("/clustering")
def clustering(req: ComparisonRequest) -> dict:
    if not req.results:
        raise HTTPException(status_code=422, detail="results is empty")
    return _maybe_register_winner(score_clustering_models(req.results), req, "clustering")
