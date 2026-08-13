"""
Load stage — writes validated, processed data into the Postgres warehouse
so the (future) FastAPI backend reads from a queryable table instead of
flat files.

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
