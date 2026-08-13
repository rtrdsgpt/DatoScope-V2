"""
Transform stage — reads from the raw zone, cleans the data with the same
logic the app uses (utils.preprocessing.clean_dataframe), and writes the
result to the processed zone. Explicit pipeline stage instead of the ad hoc
in-app sidebar calls.
"""

from __future__ import annotations

from datetime import datetime, timezone

from etl.storage import ObjectStore, new_run_id
from utils.preprocessing import clean_dataframe


def transform_raw(
    raw_bucket: str,
    raw_data_key: str,
    dataset_name: str,
    source_run_id: str,
    *,
    missing_strategy: str = "mean",
    outlier_method: str = "IQR",
    scale_method: str = "Standard",
    remove_dupes: bool = True,
    label_col: str | None = None,
    encode_categoricals: bool = False,
    categorical_encoding: str = "One-Hot",
    categorical_encoding_map: dict[str, str] | None = None,
    store: ObjectStore | None = None,
) -> dict:
    """
    `source_run_id` identifies the raw extract being transformed. The
    processed output gets its *own* fresh run_id — transform can be called
    repeatedly against the same raw extract (e.g. a user tweaking cleaning
    parameters and re-running), and each call must land as a distinct,
    separately-queryable warehouse run rather than colliding under one id.
    """
    store = store or ObjectStore()
    raw_df = store.get_dataframe(raw_bucket, raw_data_key)

    clean_df, report = clean_dataframe(
        raw_df,
        missing_strategy=missing_strategy,
        outlier_method=outlier_method,
        scale_method=scale_method,
        remove_dupes=remove_dupes,
        label_col=label_col,
        encode_categoricals=encode_categoricals,
        categorical_encoding=categorical_encoding,
        categorical_encoding_map=categorical_encoding_map,
    )

    store.ensure_zones()
    run_id = new_run_id()
    prefix = f"processed/{dataset_name}/{run_id}"
    data_key = f"{prefix}/data.parquet"
    meta_key = f"{prefix}/metadata.json"

    store.put_dataframe(store.settings.processed_bucket, data_key, clean_df)
    metadata = {
        "dataset_name": dataset_name,
        "run_id": run_id,
        "source_run_id": source_run_id,
        "transformed_at": datetime.now(timezone.utc).isoformat(),
        "source_bucket": raw_bucket,
        "source_key": raw_data_key,
        "label_col": label_col,
        "report": report,
    }
    store.put_json(store.settings.processed_bucket, meta_key, metadata)

    return {
        "bucket": store.settings.processed_bucket,
        "data_key": data_key,
        "meta_key": meta_key,
        "run_id": run_id,
        "source_run_id": source_run_id,
        "report": report,
        "df": clean_df,
    }
