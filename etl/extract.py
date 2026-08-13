"""
Extract stage — lands raw data in the S3/MinIO "raw" zone.

Covers the three sources DatoScope needs to support:
  * synthetic datasets generated in-app (`extract_generated`)
  * user-uploaded files (`extract_uploaded`)
  * a real external source, Kaggle via kagglehub (`extract_kaggle`)

Every extractor writes the same shape of output: a parquet file plus a
metadata.json sidecar under `raw/<source>/<dataset_name>/<run_id>/`, so the
transform stage doesn't need to know which extractor produced the data.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone

import pandas as pd

from data_loader import load_any_dataset
from etl.config import get_settings
from etl.storage import ObjectStore
from utils.preprocessing import parse_uploaded_bytes


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_raw(store: ObjectStore, source: str, dataset_name: str, df: pd.DataFrame, extra_meta: dict) -> dict:
    store.ensure_zones()
    run_id = _run_id()
    prefix = f"raw/{source}/{dataset_name}/{run_id}"
    data_key = f"{prefix}/data.parquet"
    meta_key = f"{prefix}/metadata.json"

    store.put_dataframe(store.settings.raw_bucket, data_key, df)
    metadata = {
        "source": source,
        "dataset_name": dataset_name,
        "run_id": run_id,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        **extra_meta,
    }
    store.put_json(store.settings.raw_bucket, meta_key, metadata)

    return {"bucket": store.settings.raw_bucket, "data_key": data_key, "meta_key": meta_key, "run_id": run_id, "metadata": metadata}


def extract_generated(df: pd.DataFrame, dataset_name: str, generator_meta: dict, store: ObjectStore | None = None) -> dict:
    """Land an in-app synthetically generated dataset in the raw zone."""
    store = store or ObjectStore()
    return _write_raw(store, "generated", dataset_name, df, {"generator_meta": generator_meta})


def extract_uploaded(filename: str, raw_bytes: bytes, dataset_name: str, store: ObjectStore | None = None) -> dict:
    """Land a user-uploaded file in the raw zone."""
    store = store or ObjectStore()
    df = parse_uploaded_bytes(filename, raw_bytes)
    return _write_raw(store, "uploaded", dataset_name, df, {"original_filename": filename})


def extract_kaggle(handle: str | None = None, dataset_name: str | None = None, store: ObjectStore | None = None) -> dict:
    """
    Land a real external dataset pulled from Kaggle in the raw zone.

    `handle` is a Kaggle dataset slug, e.g. "uciml/iris". Requires
    KAGGLE_API_TOKEN (or KAGGLE_USERNAME/KAGGLE_KEY) in the environment —
    see .env.example.
    """
    import kagglehub

    settings = get_settings()
    store = store or ObjectStore()
    handle = handle or settings.kaggle_dataset_handle
    dataset_name = dataset_name or handle.split("/")[-1]

    download_path = kagglehub.dataset_download(handle)
    csv_files = sorted(glob.glob(os.path.join(download_path, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in Kaggle dataset '{handle}' at {download_path}")

    df = load_any_dataset(csv_files[0], use_cache=False)
    return _write_raw(
        store,
        "kaggle",
        dataset_name,
        df,
        {
            "kaggle_handle": handle,
            "source_file": os.path.basename(csv_files[0]),
            "source_url": f"https://www.kaggle.com/datasets/{handle}",
        },
    )
