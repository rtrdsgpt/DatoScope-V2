"""
Entrypoint for the mlflow docker-compose service.

The warehouse Postgres container only auto-creates its own POSTGRES_DB
("datoscope") on first init, and MinIO doesn't auto-create buckets — so
this ensures the "mlflow" database and artifact bucket exist (idempotent,
safe to run on every container start) before handing off to `mlflow server`.
"""

from __future__ import annotations

import os
import subprocess

import boto3
import psycopg2
from botocore.exceptions import ClientError

PG_HOST = os.environ["MLFLOW_PG_HOST"]
PG_USER = os.environ["MLFLOW_PG_USER"]
PG_PASSWORD = os.environ["MLFLOW_PG_PASSWORD"]
PG_DB = os.environ["MLFLOW_PG_DB"]

S3_ENDPOINT_URL = os.environ["MLFLOW_S3_ENDPOINT_URL"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
BUCKET = os.environ["MLFLOW_BUCKET"]


def ensure_database() -> None:
    conn = psycopg2.connect(host=PG_HOST, user=PG_USER, password=PG_PASSWORD, dbname="postgres")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PG_DB,))
        if cur.fetchone() is None:
            cur.execute(f'CREATE DATABASE "{PG_DB}"')
    conn.close()


def ensure_bucket() -> None:
    client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    try:
        client.head_bucket(Bucket=BUCKET)
    except ClientError:
        client.create_bucket(Bucket=BUCKET)


if __name__ == "__main__":
    ensure_database()
    ensure_bucket()
    # --allowed-hosts: MLflow 3.x validates the Host header to guard against DNS
    # rebinding; the default (localhost + private IPs) doesn't include the
    # docker-compose service hostname other containers connect through.
    subprocess.run(
        [
            "mlflow", "server",
            "--backend-store-uri", f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:5432/{PG_DB}",
            "--default-artifact-root", f"s3://{BUCKET}/",
            "--host", "0.0.0.0",
            "--port", "5000",
            "--allowed-hosts", "mlflow:5000,localhost:5000,localhost:5001,127.0.0.1:5000,127.0.0.1:5001",
        ],
        check=True,
    )
