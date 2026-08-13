"""Autonomous pipeline agent endpoint (agent/pipeline_agent.py)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agent.pipeline_agent import run as run_agent
from api.schemas import AgentRunRequest

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run")
def run(req: AgentRunRequest) -> dict:
    try:
        return run_agent(req.goal, dataset_name=req.dataset_name, max_turns=req.max_turns)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent run failed: {exc}") from exc
