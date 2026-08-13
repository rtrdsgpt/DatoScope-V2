"""
DatoScope FastAPI backend — exposes dataset ingestion (via the ETL pipeline),
EDA, preprocessing, modeling, clustering, and comparison as REST endpoints,
reading from the Postgres warehouse instead of local files.

Run with: uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from api.routers import clustering, comparison, datasets, eda, modeling, models

app = FastAPI(title="DatoScope API", version="0.1.0")

app.include_router(datasets.router)
app.include_router(eda.router)
app.include_router(modeling.router)
app.include_router(clustering.router)
app.include_router(comparison.router)
app.include_router(models.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
