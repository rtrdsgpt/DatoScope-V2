"""
LLM co-pilot endpoints — grounded EDA explanations and preprocessing
recommendations (agent/copilot.py) over a warehouse-backed dataset.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agent.copilot import explain_eda, recommend_preprocessing
from api.routers.eda import _boxplot_impl, _get_df, _missing_impl, _numeric_columns, _summary_impl
from api.schemas import CopilotExplainRequest, CopilotRecommendRequest

router = APIRouter(prefix="/copilot", tags=["copilot"])


def _build_findings(dataset_name: str, run_id: str | None) -> dict:
    # Calls eda.py's plain computation functions directly (not the route
    # handlers) — those take `columns: list[str] = Query(None)`, a FastAPI
    # marker only resolved when invoked through the actual HTTP request
    # machinery; calling them as plain functions passes the Query object
    # through unresolved and breaks on the first `columns` iteration.
    df = _get_df(dataset_name, run_id)
    num_cols = _numeric_columns(df, None)
    return {
        "summary": _summary_impl(df, num_cols),
        "missing": _missing_impl(df),
        "boxplot": _boxplot_impl(df, num_cols),
    }


@router.post("/{dataset_name}/explain")
def explain(dataset_name: str, req: CopilotExplainRequest) -> dict:
    findings = _build_findings(dataset_name, req.run_id)
    try:
        result = explain_eda(req.question, findings)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Co-pilot request failed: {exc}") from exc
    return result.model_dump()


@router.post("/{dataset_name}/recommend")
def recommend(dataset_name: str, req: CopilotRecommendRequest) -> dict:
    findings = _build_findings(dataset_name, req.run_id)
    try:
        result = recommend_preprocessing(findings)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Co-pilot request failed: {exc}") from exc
    return result.model_dump()
