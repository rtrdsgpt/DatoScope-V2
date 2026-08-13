"""MLflow Model Registry endpoints — list versions, promote Staging -> Production."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.tracking import ModelPromotionError, get_model_versions, promote_to_production

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/{registered_name}/versions")
def versions(registered_name: str) -> list[dict]:
    result = get_model_versions(registered_name)
    if not result:
        raise HTTPException(status_code=404, detail=f"No registered model '{registered_name}'")
    return result


@router.post("/{registered_name}/promote")
def promote(registered_name: str) -> dict:
    try:
        return promote_to_production(registered_name)
    except ModelPromotionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
