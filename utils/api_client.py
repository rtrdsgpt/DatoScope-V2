"""
Thin HTTP client for the DatoScope FastAPI backend (api/). Streamlit talks
to the API through this module instead of calling utils.generators /
utils.preprocessing / utils.modeling directly — the API is the one place
that does ingestion (via the ETL pipeline), EDA, modeling, and clustering.

Base URL: API_BASE_URL env var, defaults to http://localhost:8000 (the
`uvicorn api.main:app` from the README's local dev instructions).
"""

from __future__ import annotations

import os

import pandas as pd
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 120  # model training can take a few seconds on larger datasets


class ApiError(RuntimeError):
    """Raised with the API's own error detail message, ready to show the user."""


def _request(method: str, path: str, **kwargs) -> dict:
    try:
        resp = requests.request(method, f"{API_BASE_URL}{path}", timeout=_TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(f"Could not reach the DatoScope API at {API_BASE_URL}. Is `uvicorn api.main:app` running?") from exc

    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(detail if isinstance(detail, str) else str(detail))
    return resp.json()


def _get(path: str, **params) -> dict:
    clean_params = {k: v for k, v in params.items() if v is not None}
    return _request("GET", path, params=clean_params)


def _post(path: str, json: dict) -> dict:
    return _request("POST", path, json=json)


# --- datasets --------------------------------------------------------------

def generate_dataset(**payload) -> dict:
    return _post("/datasets/generate", payload)


def upload_dataset(filename: str, raw_bytes: bytes, dataset_name: str | None = None) -> dict:
    data = {"dataset_name": dataset_name} if dataset_name else {}
    try:
        resp = requests.post(
            f"{API_BASE_URL}/datasets/upload",
            data=data,
            files={"file": (filename, raw_bytes)},
            timeout=_TIMEOUT,
        )
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(f"Could not reach the DatoScope API at {API_BASE_URL}. Is `uvicorn api.main:app` running?") from exc
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(detail if isinstance(detail, str) else str(detail))
    return resp.json()


def kaggle_dataset(handle: str, dataset_name: str | None = None) -> dict:
    return _post("/datasets/kaggle", {"handle": handle, "dataset_name": dataset_name})


def clean_dataset(dataset_name: str, **payload) -> dict:
    return _post(f"/datasets/{dataset_name}/clean", payload)


def list_datasets() -> list[dict]:
    return _get("/datasets")


def list_runs(dataset_name: str) -> list[dict]:
    return _get(f"/datasets/{dataset_name}/runs")


def get_data(dataset_name: str, run_id: str | None = None, limit: int = 5000) -> pd.DataFrame:
    result = _get(f"/datasets/{dataset_name}/data", run_id=run_id, limit=limit)
    return pd.DataFrame.from_records(result["records"], columns=result["columns"])


def get_raw(dataset_name: str, run_id: str | None = None, limit: int = 5000) -> pd.DataFrame:
    result = _get(f"/datasets/{dataset_name}/raw", run_id=run_id, limit=limit)
    return pd.DataFrame.from_records(result["records"], columns=result["columns"])


# --- eda ---------------------------------------------------------------

def eda_summary(dataset_name: str, run_id: str | None = None, columns: list[str] | None = None) -> dict:
    return _get(f"/eda/{dataset_name}/summary", run_id=run_id, columns=columns)


def eda_missing(dataset_name: str, run_id: str | None = None) -> dict:
    return _get(f"/eda/{dataset_name}/missing", run_id=run_id)


def eda_distributions(dataset_name: str, run_id: str | None = None, columns: list[str] | None = None, bins: int = 40) -> dict:
    return _get(f"/eda/{dataset_name}/distributions", run_id=run_id, columns=columns, bins=bins)


def eda_boxplot(dataset_name: str, run_id: str | None = None, columns: list[str] | None = None) -> dict:
    return _get(f"/eda/{dataset_name}/boxplot", run_id=run_id, columns=columns)


def eda_qq(dataset_name: str, run_id: str | None = None, columns: list[str] | None = None) -> dict:
    return _get(f"/eda/{dataset_name}/qq", run_id=run_id, columns=columns)


def eda_correlation(dataset_name: str, run_id: str | None = None, columns: list[str] | None = None, top_n: int = 10) -> dict:
    return _get(f"/eda/{dataset_name}/correlation", run_id=run_id, columns=columns, top_n=top_n)


def eda_variance(dataset_name: str, run_id: str | None = None, columns: list[str] | None = None) -> dict:
    return _get(f"/eda/{dataset_name}/variance", run_id=run_id, columns=columns)


# --- modeling ---------------------------------------------------------

def train_regression(dataset_name: str, **payload) -> dict:
    return _post(f"/modeling/{dataset_name}/regression", payload)


def train_classification(dataset_name: str, **payload) -> dict:
    return _post(f"/modeling/{dataset_name}/classification", payload)


def download_model(model_id: str) -> bytes:
    try:
        resp = requests.get(f"{API_BASE_URL}/modeling/download/{model_id}", timeout=_TIMEOUT)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(f"Could not reach the DatoScope API at {API_BASE_URL}.") from exc
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiError(detail if isinstance(detail, str) else str(detail))
    return resp.content


# --- clustering ---------------------------------------------------------

def run_clustering(dataset_name: str, **payload) -> dict:
    return _post(f"/clustering/{dataset_name}", payload)


# --- comparison ---------------------------------------------------------

def compare_regression(results: dict) -> dict:
    return _post("/comparison/regression", {"results": results})


def compare_classification(results: dict) -> dict:
    return _post("/comparison/classification", {"results": results})


def compare_clustering(results: dict) -> dict:
    return _post("/comparison/clustering", {"results": results})
