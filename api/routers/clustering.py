"""Clustering endpoint — train/evaluate KMeans/DBSCAN/Hierarchical, plus a 2D projection for plotting."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import ClusteringRequest
from api.serialization import serialize_clustering_results
from api.tracking import log_model_run
from etl.load import DatasetNotFoundError, get_dataset
from utils.modeling import projection_for_plot, run_clustering_models

router = APIRouter(prefix="/clustering", tags=["clustering"])

_METRIC_KEYS = ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz", "FM_Score", "Rand_Index"]


@router.post("/{dataset_name}")
def cluster(dataset_name: str, req: ClusteringRequest) -> dict:
    try:
        df = get_dataset(dataset_name, req.run_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    missing = [c for c in req.features if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Column(s) not found: {missing}")
    if len(req.features) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 feature columns")
    if req.ground_truth_col and req.ground_truth_col not in df.columns:
        raise HTTPException(status_code=422, detail=f"ground_truth_col '{req.ground_truth_col}' not found")

    ground_truth = df[req.ground_truth_col] if req.ground_truth_col else None
    result = run_clustering_models(
        df,
        features=req.features,
        run_km=req.run_km,
        k_val=req.k_val,
        run_db=req.run_db,
        db_eps=req.db_eps,
        db_min=req.db_min,
        run_hc=req.run_hc,
        hc_k=req.hc_k,
        hc_link=req.hc_link,
        ground_truth=ground_truth,
    )
    if not result["results"]:
        raise HTTPException(status_code=422, detail="No algorithms selected (set at least one of run_km/run_db/run_hc)")

    out = serialize_clustering_results(result)

    params = {
        "k_val": req.k_val,
        "db_eps": req.db_eps,
        "db_min": req.db_min,
        "hc_k": req.hc_k,
        "hc_link": req.hc_link,
        "features": ",".join(req.features),
    }
    for name, row in result["results"].items():
        try:
            run_id = log_model_run(
                dataset_name=dataset_name,
                task="clustering",
                model_name=name,
                model=row["model"],
                params=params,
                metrics={k: out["models"][name][k] for k in _METRIC_KEYS if out["models"][name].get(k) is not None},
            )
            out["models"][name]["mlflow_run_id"] = run_id
        except Exception as exc:
            out["models"][name]["mlflow_run_id"] = None
            out["models"][name]["mlflow_error"] = str(exc)

    coords, x_label, y_label = projection_for_plot(df, req.features)
    out["projection"] = {"x": coords[:, 0].tolist(), "y": coords[:, 1].tolist(), "x_label": x_label, "y_label": y_label}
    return out
