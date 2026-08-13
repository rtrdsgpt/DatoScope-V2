"""
Comparison endpoints — stateless winner-selection over metrics a client
already has (e.g. straight from a /modeling or /clustering response).
Mirrors pages/4_Comparison.py's scoring logic (utils/comparison.py).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import ComparisonRequest
from utils.comparison import score_classification_models, score_clustering_models, score_regression_models

router = APIRouter(prefix="/comparison", tags=["comparison"])


@router.post("/regression")
def regression(req: ComparisonRequest) -> dict:
    if not req.results:
        raise HTTPException(status_code=422, detail="results is empty")
    return score_regression_models(req.results)


@router.post("/classification")
def classification(req: ComparisonRequest) -> dict:
    if not req.results:
        raise HTTPException(status_code=422, detail="results is empty")
    return score_classification_models(req.results)


@router.post("/clustering")
def clustering(req: ComparisonRequest) -> dict:
    if not req.results:
        raise HTTPException(status_code=422, detail="results is empty")
    return score_clustering_models(req.results)
