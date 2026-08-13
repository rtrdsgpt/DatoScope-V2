"""
Orchestration-agnostic pipeline stages — thin functions that pass
bucket/key references (not DataFrames) between stages, so the same
functions work whether they're called directly, from a script, or as
Airflow PythonOperators (see airflow/dags/datoscope_etl_dag.py).
"""

from __future__ import annotations

from typing import Any

from etl.extract import extract_generated, extract_kaggle, extract_uploaded
from etl.load import load_to_warehouse
from etl.storage import ObjectStore
from etl.transform import transform_raw
from etl.validate import validate_dataframe


def run_extract(source: str, *, dataset_name: str | None = None, store: ObjectStore | None = None, **kwargs) -> dict:
    store = store or ObjectStore()
    if source == "generated":
        return extract_generated(kwargs["df"], dataset_name or "generated_dataset", kwargs.get("generator_meta", {}), store)
    if source == "uploaded":
        return extract_uploaded(kwargs["filename"], kwargs["raw_bytes"], dataset_name or kwargs["filename"], store)
    if source == "kaggle":
        return extract_kaggle(kwargs.get("handle"), dataset_name, store)
    raise ValueError(f"Unknown extract source: {source}")


def run_transform(raw_ref: dict, *, transform_kwargs: dict[str, Any] | None = None, store: ObjectStore | None = None) -> dict:
    store = store or ObjectStore()
    return transform_raw(
        raw_ref["bucket"],
        raw_ref["data_key"],
        raw_ref["metadata"]["dataset_name"],
        raw_ref["run_id"],
        store=store,
        **(transform_kwargs or {}),
    )


def run_validate(processed_ref: dict, *, validate_kwargs: dict[str, Any] | None = None, store: ObjectStore | None = None) -> dict:
    store = store or ObjectStore()
    df = processed_ref.get("df")
    if df is None:
        df = store.get_dataframe(processed_ref["bucket"], processed_ref["data_key"])
    dataset_name = processed_ref.get("dataset_name") or store.get_json(processed_ref["bucket"], processed_ref["meta_key"])["dataset_name"]
    return validate_dataframe(df, dataset_name=dataset_name, **(validate_kwargs or {}))


def run_load(processed_ref: dict, *, source: str = "unknown", store: ObjectStore | None = None) -> dict:
    store = store or ObjectStore()
    df = processed_ref.get("df")
    if df is None:
        df = store.get_dataframe(processed_ref["bucket"], processed_ref["data_key"])
    meta = store.get_json(processed_ref["bucket"], processed_ref["meta_key"])
    return load_to_warehouse(df, dataset_name=meta["dataset_name"], run_id=meta["run_id"], source=source)


def run_pipeline(
    source: str,
    *,
    dataset_name: str | None = None,
    extract_kwargs: dict[str, Any] | None = None,
    transform_kwargs: dict[str, Any] | None = None,
    validate_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Run extract -> transform -> validate -> load end to end, in-process."""
    store = ObjectStore()

    raw_ref = run_extract(source, dataset_name=dataset_name, store=store, **(extract_kwargs or {}))
    processed_ref = run_transform(raw_ref, transform_kwargs=transform_kwargs, store=store)
    processed_ref["dataset_name"] = raw_ref["metadata"]["dataset_name"]
    validation_report = run_validate(processed_ref, validate_kwargs=validate_kwargs, store=store)
    load_summary = run_load(processed_ref, source=source, store=store)

    return {
        "raw": raw_ref,
        "processed": {k: v for k, v in processed_ref.items() if k != "df"},
        "validation": validation_report,
        "load": load_summary,
    }
