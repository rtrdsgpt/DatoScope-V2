"""
EDA endpoints — pure computation over a warehouse-backed dataset (summary
stats, missing values, distributions, box-plot/outlier stats, Q-Q plot
points, correlation, variance ranking). Mirrors pages/1_EDA.py's numbers;
plotting itself stays a Streamlit/Plotly concern, these endpoints return
the underlying data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from scipy import stats

from etl.load import DatasetNotFoundError, get_dataset

router = APIRouter(prefix="/eda", tags=["eda"])


def _get_df(dataset_name: str, run_id: str | None) -> pd.DataFrame:
    try:
        return get_dataset(dataset_name, run_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _numeric_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if columns:
        missing = [c for c in columns if c not in num_cols]
        if missing:
            raise HTTPException(status_code=422, detail=f"Not numeric or not found: {missing}")
        return columns
    return num_cols


def _summary_impl(df: pd.DataFrame, num_cols: list[str]) -> dict:
    if not num_cols:
        raise HTTPException(status_code=422, detail="No numeric columns found")
    stats_df = df[num_cols].describe().T
    stats_df["skewness"] = df[num_cols].skew().round(3)
    stats_df["kurtosis"] = df[num_cols].kurtosis().round(3)
    return {"columns": stats_df.reset_index(names="feature").to_dict(orient="records")}


@router.get("/{dataset_name}/summary")
def summary(dataset_name: str, run_id: str | None = None, columns: list[str] | None = Query(None)) -> dict:
    df = _get_df(dataset_name, run_id)
    return _summary_impl(df, _numeric_columns(df, columns))


def _missing_impl(df: pd.DataFrame) -> dict:
    miss = (df.isnull().mean() * 100).round(3)
    miss = miss[miss > 0].sort_values(ascending=False)
    return {"missing_pct": miss.to_dict()}


@router.get("/{dataset_name}/missing")
def missing(dataset_name: str, run_id: str | None = None) -> dict:
    return _missing_impl(_get_df(dataset_name, run_id))


@router.get("/{dataset_name}/distributions")
def distributions(dataset_name: str, run_id: str | None = None, columns: list[str] | None = Query(None), bins: int = 40) -> dict:
    df = _get_df(dataset_name, run_id)
    num_cols = _numeric_columns(df, columns)
    out = {}
    for col in num_cols:
        counts, edges = np.histogram(df[col].dropna(), bins=bins)
        out[col] = {"counts": counts.tolist(), "bin_edges": edges.tolist()}
    return {"distributions": out}


def _boxplot_impl(df: pd.DataFrame, num_cols: list[str]) -> dict:
    out = {}
    for col in num_cols:
        series = df[col].dropna()
        q1, median, q3 = series.quantile(0.25), series.quantile(0.5), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        outlier_count = int(outlier_mask.sum())
        out[col] = {
            "q1": round(q1, 4),
            "median": round(median, 4),
            "q3": round(q3, 4),
            "mean": round(series.mean(), 4),
            "std": round(series.std(), 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": outlier_count,
            "outlier_pct": round((outlier_count / len(series)) * 100, 2) if len(series) else 0.0,
        }
    return {"boxplot": out}


@router.get("/{dataset_name}/boxplot")
def boxplot(dataset_name: str, run_id: str | None = None, columns: list[str] | None = Query(None)) -> dict:
    df = _get_df(dataset_name, run_id)
    return _boxplot_impl(df, _numeric_columns(df, columns))


@router.get("/{dataset_name}/qq")
def qq(dataset_name: str, run_id: str | None = None, columns: list[str] | None = Query(None)) -> dict:
    df = _get_df(dataset_name, run_id)
    num_cols = _numeric_columns(df, columns)
    out = {}
    for col in num_cols:
        series = df[col].dropna()
        osm, osr = stats.probplot(series, dist="norm", fit=False)
        slope, intercept, _ = stats.probplot(series, dist="norm", fit=True)[1]
        out[col] = {"osm": np.asarray(osm).tolist(), "osr": np.asarray(osr).tolist(), "slope": float(slope), "intercept": float(intercept)}
    return {"qq": out}


@router.get("/{dataset_name}/correlation")
def correlation(dataset_name: str, run_id: str | None = None, columns: list[str] | None = Query(None), top_n: int = 10) -> dict:
    df = _get_df(dataset_name, run_id)
    num_cols = _numeric_columns(df, columns)
    if len(num_cols) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 numeric columns")
    corr = df[num_cols].corr()
    corr_flat = corr.mask(np.triu(np.ones(corr.shape, dtype=bool))).stack().reset_index()
    corr_flat.columns = ["feature_a", "feature_b", "correlation"]
    corr_flat["abs"] = corr_flat["correlation"].abs()
    top = corr_flat.nlargest(top_n, "abs").drop(columns="abs")
    return {"matrix": corr.round(4).to_dict(), "top_pairs": top.to_dict(orient="records")}


@router.get("/{dataset_name}/variance")
def variance(dataset_name: str, run_id: str | None = None, columns: list[str] | None = Query(None)) -> dict:
    df = _get_df(dataset_name, run_id)
    num_cols = _numeric_columns(df, columns)
    var = df[num_cols].var().sort_values(ascending=False)
    threshold = max(var.max() * 0.05, 0.01) if len(var) else 0.0
    out = []
    for feature, value in var.items():
        if value < threshold:
            interpretation = "Low spread"
        elif value > var.median():
            interpretation = "High spread"
        else:
            interpretation = "Moderate spread"
        out.append({"feature": feature, "variance": round(float(value), 4), "interpretation": interpretation})
    return {"variance": out}
