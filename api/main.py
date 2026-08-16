"""
DatoScope FastAPI backend — exposes dataset ingestion (via the ETL pipeline),
EDA, preprocessing, modeling, clustering, and comparison as REST endpoints,
reading from the Postgres warehouse instead of local files.

Run with: uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.routers import agent, clustering, comparison, copilot, datasets, eda, modeling, models

app = FastAPI(title="DatoScope API", version="0.1.0")

app.include_router(datasets.router)
app.include_router(eda.router)
app.include_router(modeling.router)
app.include_router(clustering.router)
app.include_router(comparison.router)
app.include_router(models.router)
app.include_router(copilot.router)
app.include_router(agent.router)

# Structured metrics for the non-LLM pipeline stages (todo.md section 8) —
# request rate/latency/in-progress per route out of the box, exposed at
# /metrics for Prometheus to scrape. The LLM/agent layer is traced by
# Langfuse instead (todo.md section 8), not stretched over this too.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
