"""Supervised modeling endpoints — train/evaluate, and download a trained model."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from api.schemas import ClassificationRequest, RegressionRequest
from api.serialization import MODEL_REGISTRY, serialize_classification_results, serialize_regression_results
from etl.load import DatasetNotFoundError, get_dataset
from utils.modeling import export_model_bytes, run_classification_models, run_regression_models

router = APIRouter(prefix="/modeling", tags=["modeling"])


def _get_df(dataset_name: str, run_id: str | None):
    try:
        return get_dataset(dataset_name, run_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{dataset_name}/regression")
def regression(dataset_name: str, req: RegressionRequest) -> dict:
    df = _get_df(dataset_name, req.run_id)
    missing = [c for c in [*req.features, req.target_col] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Column(s) not found: {missing}")

    results = run_regression_models(
        df,
        None,
        features=req.features,
        target_col=req.target_col,
        test_size=req.test_size,
        cv_folds=req.cv_folds,
        random_seed=req.random_seed,
        run_lr=req.run_lr,
        run_ridge=req.run_ridge,
        ridge_alpha=req.ridge_alpha,
        run_lasso=req.run_lasso,
        lasso_alpha=req.lasso_alpha,
    )
    if not results:
        raise HTTPException(status_code=422, detail="No models selected (set at least one of run_lr/run_ridge/run_lasso)")
    return serialize_regression_results(results, dataset_name=dataset_name, target_col=req.target_col)


@router.post("/{dataset_name}/classification")
def classification(dataset_name: str, req: ClassificationRequest) -> dict:
    df = _get_df(dataset_name, req.run_id)
    missing = [c for c in [*req.features, req.target_col] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Column(s) not found: {missing}")

    results = run_classification_models(
        df,
        None,
        features=req.features,
        target_col=req.target_col,
        test_size=req.test_size,
        cv_folds=req.cv_folds,
        random_seed=req.random_seed,
        run_logreg=req.run_logreg,
        run_rf=req.run_rf,
        rf_estimators=req.rf_estimators,
        rf_max_depth=req.rf_max_depth,
        rf_min_samples_split=req.rf_min_samples_split,
        rf_min_samples_leaf=req.rf_min_samples_leaf,
        rf_max_features=req.rf_max_features,
        rf_bootstrap=req.rf_bootstrap,
        run_knn=req.run_knn,
        knn_neighbors=req.knn_neighbors,
    )
    if not results:
        raise HTTPException(status_code=422, detail="No models selected (set at least one of run_logreg/run_rf/run_knn)")
    return serialize_classification_results(results, dataset_name=dataset_name, target_col=req.target_col)


@router.get("/download/{model_id}")
def download(model_id: str) -> Response:
    entry = MODEL_REGISTRY.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No model with id '{model_id}' (models are kept in memory and don't survive an API restart)")
    payload = export_model_bytes(entry["model"], features=entry["features"], target=entry["target"], task_type=entry["task_type"])
    return Response(content=payload, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{model_id}.pkl"'})
