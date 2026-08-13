"""
Dataset ingestion (extract) and warehouse access (list/data) endpoints.

Ingestion is split into two steps mirroring the ETL pipeline stages:
  1. extract  (generate / upload / kaggle) -> lands raw data, returns a run_id
  2. clean    -> transform + validate + load, using that run_id's raw data

Splitting them lets a client preview a raw extract before committing to a
particular cleaning configuration, same as the current Streamlit sidebar's
"pick options, see a live estimate, then click Clean & Preprocess" flow.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.schemas import CleanRequest, GenerateDatasetRequest, KaggleDatasetRequest
from etl.extract import RawRunNotFoundError, extract_generated, extract_kaggle, extract_uploaded, find_raw_run
from etl.load import DatasetNotFoundError, get_dataset, list_datasets, list_runs, load_to_warehouse
from etl.storage import ObjectStore
from etl.transform import transform_raw
from etl.validate import DataQualityError, validate_dataframe
from utils.generators import generate_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _raw_preview(raw_ref: dict) -> dict:
    return {
        "dataset_name": raw_ref["metadata"]["dataset_name"],
        "run_id": raw_ref["run_id"],
        "source": raw_ref["metadata"]["source"],
        "n_rows": raw_ref["metadata"]["n_rows"],
        "n_cols": raw_ref["metadata"]["n_cols"],
        "columns": raw_ref["metadata"]["columns"],
    }


@router.post("/generate")
def generate(req: GenerateDatasetRequest) -> dict:
    df, meta = generate_dataset(
        req.task_type,
        req.dataset_type,
        n_samples=req.n_samples,
        noise=req.noise,
        n_clusters=req.n_clusters,
        random_seed=req.random_seed,
        n_features=req.n_features,
        n_informative=req.n_informative,
        target_name=req.target_name,
    )
    dataset_name = req.dataset_name or f"{req.task_type.lower()}_{req.dataset_type.lower().replace(' ', '_')}"
    raw_ref = extract_generated(df, dataset_name, meta)
    return {**_raw_preview(raw_ref), "generator_meta": meta}


@router.post("/upload")
async def upload(dataset_name: str | None = Form(None), file: UploadFile = File(...)) -> dict:
    raw_bytes = await file.read()
    name = dataset_name or file.filename
    try:
        raw_ref = extract_uploaded(file.filename, raw_bytes, name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _raw_preview(raw_ref)


@router.post("/kaggle")
def kaggle(req: KaggleDatasetRequest) -> dict:
    try:
        raw_ref = extract_kaggle(req.handle, req.dataset_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {**_raw_preview(raw_ref), "kaggle_handle": req.handle}


@router.post("/{dataset_name}/clean")
def clean(dataset_name: str, req: CleanRequest) -> dict:
    store = ObjectStore()
    try:
        raw_ref = find_raw_run(dataset_name, req.run_id, store=store)
    except RawRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    processed = transform_raw(
        raw_ref["bucket"],
        raw_ref["data_key"],
        dataset_name,
        raw_ref["run_id"],
        missing_strategy=req.missing_strategy,
        outlier_method=req.outlier_method,
        scale_method=req.scale_method,
        remove_dupes=req.remove_dupes,
        label_col=req.label_col,
        encode_categoricals=req.encode_categoricals,
        categorical_encoding=req.categorical_encoding,
        categorical_encoding_map=req.categorical_encoding_map,
        store=store,
    )

    try:
        validation = validate_dataframe(processed["df"], dataset_name=dataset_name, required_columns=[req.label_col] if req.label_col else None)
    except DataQualityError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), "report": exc.report}) from exc

    load_summary = load_to_warehouse(
        processed["df"],
        dataset_name=dataset_name,
        run_id=processed["run_id"],
        source=raw_ref["metadata"]["source"],
    )

    return {
        "dataset_name": dataset_name,
        "raw_run_id": raw_ref["run_id"],
        "run_id": processed["run_id"],
        "report": processed["report"],
        "validation": validation,
        "load": load_summary,
    }


@router.get("")
def list_all() -> list[dict]:
    return list_datasets()


@router.get("/{dataset_name}/runs")
def runs(dataset_name: str) -> list[dict]:
    result = list_runs(dataset_name)
    if not result:
        raise HTTPException(status_code=404, detail=f"No warehouse data for dataset '{dataset_name}'")
    return result


@router.get("/{dataset_name}/data")
def data(dataset_name: str, run_id: str | None = None, limit: int = 500) -> dict:
    try:
        df = get_dataset(dataset_name, run_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "dataset_name": dataset_name,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "records": df.head(limit).to_dict(orient="records"),
    }
