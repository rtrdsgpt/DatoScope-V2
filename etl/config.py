"""
Environment-driven settings for the ETL pipeline.

Every value has a docker-compose-friendly local default so the pipeline runs
out of the box against the services in docker-compose.yml. Pointing this at
real AWS S3 / RDS in production is a matter of overriding the env vars (see
section 5 of todo.md) — no code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Object storage (MinIO locally, S3 in AWS — same boto3 client either way)
    s3_endpoint_url: str | None = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "datoscope")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "datoscope123")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")
    raw_bucket: str = os.getenv("RAW_BUCKET", "datoscope-raw")
    processed_bucket: str = os.getenv("PROCESSED_BUCKET", "datoscope-processed")

    # Warehouse (Postgres locally via docker-compose, RDS in AWS)
    warehouse_dsn: str = os.getenv(
        "WAREHOUSE_DSN", "postgresql+psycopg2://datoscope:datoscope123@localhost:5433/datoscope"
    )

    # Kaggle (kagglehub reads KAGGLE_API_TOKEN itself; kept here for discoverability)
    kaggle_dataset_handle: str = os.getenv("KAGGLE_DATASET_HANDLE", "uciml/iris")

    # Data quality
    max_null_fraction: float = float(os.getenv("MAX_NULL_FRACTION", "0.5"))


def get_settings() -> Settings:
    return Settings()
