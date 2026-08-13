"""
Load stage — writes validated, processed data into the Postgres warehouse,
and the read-back functions the FastAPI backend (api/) uses instead of
touching flat files.

Each dataset gets its own table (schemas vary too widely across generated /
uploaded / Kaggle datasets to share one wide table); a shared `etl_runs`
registry table indexes every run so a consumer can discover what's in the
warehouse without knowing table names up front.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from etl.config import Settings, get_settings

_engine: Engine | None = None

_REGISTRY_TABLE = "etl_runs"

_CREATE_REGISTRY_SQL = f"""
CREATE TABLE IF NOT EXISTS {_REGISTRY_TABLE} (
    dataset_name TEXT NOT NULL,
    run_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    source TEXT,
    n_rows INTEGER NOT NULL,
    n_cols INTEGER NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (dataset_name, run_id)
)
"""


def get_engine(settings: Settings | None = None) -> Engine:
    global _engine
    if _engine is None:
        settings = settings or get_settings()
        _engine = create_engine(settings.warehouse_dsn)
    return _engine


def _sanitize_table_name(dataset_name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]", "_", dataset_name.lower()).strip("_")
    return f"ds_{slug}"


def load_to_warehouse(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    run_id: str,
    source: str = "unknown",
    table_name: str | None = None,
    engine: Engine | None = None,
) -> dict:
    """
    Append `df` to a per-dataset warehouse table, tagged with run_id so
    multiple pipeline runs coexist, and index the run in `etl_runs`.
    """
    engine = engine or get_engine()
    table_name = table_name or _sanitize_table_name(dataset_name)

    out = df.copy()
    out.insert(0, "_run_id", run_id)
    out.insert(1, "_loaded_at", datetime.now(timezone.utc))

    with engine.begin() as conn:
        conn.execute(text(_CREATE_REGISTRY_SQL))
        out.to_sql(table_name, conn, if_exists="append", index=False)
        conn.execute(
            text(
                f"""
                INSERT INTO {_REGISTRY_TABLE}
                    (dataset_name, run_id, table_name, source, n_rows, n_cols, loaded_at)
                VALUES
                    (:dataset_name, :run_id, :table_name, :source, :n_rows, :n_cols, :loaded_at)
                ON CONFLICT (dataset_name, run_id) DO UPDATE SET
                    table_name = EXCLUDED.table_name,
                    source = EXCLUDED.source,
                    n_rows = EXCLUDED.n_rows,
                    n_cols = EXCLUDED.n_cols,
                    loaded_at = EXCLUDED.loaded_at
                """
            ),
            {
                "dataset_name": dataset_name,
                "run_id": run_id,
                "table_name": table_name,
                "source": source,
                "n_rows": len(out),
                "n_cols": len(df.columns),
                "loaded_at": datetime.now(timezone.utc),
            },
        )

    return {"table_name": table_name, "dataset_name": dataset_name, "run_id": run_id, "n_rows": len(out)}


class DatasetNotFoundError(Exception):
    pass


def list_datasets(engine: Engine | None = None) -> list[dict]:
    """One row per dataset, summarizing its most recent run."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text(_CREATE_REGISTRY_SQL))
        rows = conn.execute(
            text(
                f"""
                SELECT DISTINCT ON (dataset_name)
                    dataset_name, run_id, table_name, source, n_rows, n_cols, loaded_at
                FROM {_REGISTRY_TABLE}
                ORDER BY dataset_name, loaded_at DESC
                """
            )
        ).mappings().all()
    return [dict(r) for r in rows]


def list_runs(dataset_name: str, engine: Engine | None = None) -> list[dict]:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(text(_CREATE_REGISTRY_SQL))
        rows = conn.execute(
            text(
                f"""
                SELECT dataset_name, run_id, table_name, source, n_rows, n_cols, loaded_at
                FROM {_REGISTRY_TABLE}
                WHERE dataset_name = :dataset_name
                ORDER BY loaded_at DESC
                """
            ),
            {"dataset_name": dataset_name},
        ).mappings().all()
    return [dict(r) for r in rows]


def get_run(dataset_name: str, run_id: str | None = None, engine: Engine | None = None) -> dict:
    """Look up a run's registry row; falls back to the most recent run when run_id is omitted."""
    runs = list_runs(dataset_name, engine=engine)
    if not runs:
        raise DatasetNotFoundError(f"No warehouse data for dataset '{dataset_name}'")
    if run_id is None:
        return runs[0]
    for run in runs:
        if run["run_id"] == run_id:
            return run
    raise DatasetNotFoundError(f"No run '{run_id}' for dataset '{dataset_name}'")


def get_dataset(dataset_name: str, run_id: str | None = None, engine: Engine | None = None) -> pd.DataFrame:
    """Read a dataset's data back out of its warehouse table (latest run by default)."""
    engine = engine or get_engine()
    run = get_run(dataset_name, run_id, engine=engine)
    with engine.begin() as conn:
        df = pd.read_sql(
            text(f'SELECT * FROM "{run["table_name"]}" WHERE _run_id = :run_id'),
            conn,
            params={"run_id": run["run_id"]},
        )
    return df.drop(columns=["_run_id", "_loaded_at"])
