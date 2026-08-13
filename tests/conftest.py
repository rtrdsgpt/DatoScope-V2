"""Shared pytest fixtures."""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, 100),
            "x2": rng.normal(5, 2, 100),
            "cat": rng.choice(["a", "b", "c"], 100),
            "target": rng.integers(0, 2, 100),
        }
    )


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Nulls, duplicate rows, and a handful of extreme outliers — exercises every cleaning step."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, 200),
            "x2": rng.normal(5, 2, 200),
            "cat": rng.choice(["a", "b", None], 200),
            "target": rng.integers(0, 2, 200),
        }
    )
    df.loc[0:9, "x1"] = np.nan
    df.loc[10:14, "x2"] = np.nan
    df.loc[20:24, "x1"] = rng.normal(0, 1, 5) * 100  # outliers
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate row
    return df


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture
def moto_store():
    """An ObjectStore backed by moto's mocked S3 (not MinIO) — no live services needed.

    moto only intercepts the default AWS endpoint pattern, not an arbitrary
    endpoint_url like MinIO's, so this builds Settings with
    s3_endpoint_url=None specifically for the mock.
    """
    from moto import mock_aws

    from etl.config import Settings
    from etl.storage import ObjectStore

    with mock_aws():
        settings = Settings(
            s3_endpoint_url=None,
            s3_access_key="testing",
            s3_secret_key="testing",
            s3_region="us-east-1",
            raw_bucket="test-raw",
            processed_bucket="test-processed",
        )
        store = ObjectStore(settings)
        store.ensure_zones()
        yield store


@pytest.fixture
def warehouse_engine():
    """A real SQLAlchemy engine against the docker-compose warehouse Postgres. Skips if not up."""
    from etl.config import get_settings

    settings = get_settings()
    parsed = urlparse(settings.warehouse_dsn.replace("postgresql+psycopg2", "postgresql"))
    if not _port_open(parsed.hostname or "localhost", parsed.port or 5432):
        pytest.skip("warehouse Postgres not reachable — run `docker compose up -d warehouse`")

    from sqlalchemy import create_engine, text

    engine = create_engine(settings.warehouse_dsn)
    yield engine

    # Integration tests use randomly-suffixed "pytest_*" dataset names precisely
    # so repeated/parallel runs don't collide; clean up what they created so the
    # warehouse doesn't accumulate test debris across runs. Guarded because a
    # test run in isolation may finish before etl_runs ever gets created.
    try:
        with engine.begin() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT table_name FROM etl_runs WHERE dataset_name LIKE 'pytest_%'")
            ).fetchall()
            for (table_name,) in rows:
                conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
            conn.execute(text("DELETE FROM etl_runs WHERE dataset_name LIKE 'pytest_%'"))
    except Exception:
        pass
    engine.dispose()


@pytest.fixture
def minio_ready():
    """Skips the test if MinIO isn't reachable (integration tests against real object storage)."""
    from etl.config import get_settings

    settings = get_settings()
    parsed = urlparse(settings.s3_endpoint_url or "")
    if not _port_open(parsed.hostname or "localhost", parsed.port or 9000):
        pytest.skip("MinIO not reachable — run `docker compose up -d minio`")
