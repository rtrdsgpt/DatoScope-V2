"""
Airflow DAG orchestrating DatoScope's extract -> transform -> validate ->
load pipeline as 4 tasks, each a thin PythonOperator calling the *same*
etl.extract / etl.transform / etl.validate / etl.load functions the
Streamlit app and offline scripts use — one pipeline implementation, driven
by three different callers, not a parallel Airflow-only copy.

Intermediate state is handed between tasks via XCom as small dicts of
bucket/key references (never DataFrames — Airflow's XCom backend is not
meant to carry non-trivial payloads); each task re-reads the actual data
from the raw/processed zone by key.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

from etl.pipeline import run_extract, run_load, run_transform, run_validate
from etl.validate import DataQualityError

default_args = {
    "owner": "datoscope",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _kaggle_handle() -> str:
    return Variable.get("datoscope_kaggle_handle", default_var="uciml/iris")


def _dataset_name() -> str:
    return Variable.get("datoscope_dataset_name", default_var="iris")


def extract(ti, **context) -> None:
    raw_ref = run_extract("kaggle", dataset_name=_dataset_name(), handle=_kaggle_handle())
    ti.xcom_push(
        key="raw_ref",
        value={
            "bucket": raw_ref["bucket"],
            "data_key": raw_ref["data_key"],
            "meta_key": raw_ref["meta_key"],
            "run_id": raw_ref["run_id"],
            "dataset_name": raw_ref["metadata"]["dataset_name"],
        },
    )


def transform(ti, **context) -> None:
    raw_ref = ti.xcom_pull(key="raw_ref", task_ids="extract")
    raw_ref = {**raw_ref, "metadata": {"dataset_name": raw_ref["dataset_name"]}}

    processed_ref = run_transform(raw_ref)
    ti.xcom_push(
        key="processed_ref",
        value={
            "bucket": processed_ref["bucket"],
            "data_key": processed_ref["data_key"],
            "meta_key": processed_ref["meta_key"],
            "run_id": processed_ref["run_id"],
            "dataset_name": raw_ref["metadata"]["dataset_name"],
        },
    )


def validate(ti, **context) -> None:
    processed_ref = ti.xcom_pull(key="processed_ref", task_ids="transform")
    try:
        report = run_validate(processed_ref)
    except DataQualityError as exc:
        ti.xcom_push(key="validation_report", value=exc.report)
        raise
    ti.xcom_push(key="validation_report", value=report)


def load(ti, **context) -> None:
    processed_ref = ti.xcom_pull(key="processed_ref", task_ids="transform")
    summary = run_load(processed_ref, source="kaggle")
    ti.xcom_push(key="load_summary", value=summary)


with DAG(
    dag_id="datoscope_etl_dag",
    description="Extract (Kaggle) -> transform -> validate -> load into the DatoScope warehouse",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["datoscope", "etl"],
) as dag:
    t1 = PythonOperator(task_id="extract", python_callable=extract)
    t2 = PythonOperator(task_id="transform", python_callable=transform)
    t3 = PythonOperator(task_id="validate", python_callable=validate)
    t4 = PythonOperator(task_id="load", python_callable=load)

    t1 >> t2 >> t3 >> t4
