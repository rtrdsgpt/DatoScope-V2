"""
Thin ObjectStore over boto3's S3 client.

Talks to MinIO locally (S3-compatible, see docker-compose.yml) and to real
AWS S3 in production — same client, only the endpoint/credentials change
(section 5 of todo.md provisions the real bucket via Terraform).
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from etl.config import Settings, get_settings


def new_run_id() -> str:
    """Sortable UTC timestamp string — lexicographic order matches chronological order."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class ObjectStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = boto3.client(
            "s3",
            endpoint_url=self.settings.s3_endpoint_url,
            aws_access_key_id=self.settings.s3_access_key,
            aws_secret_access_key=self.settings.s3_secret_key,
            region_name=self.settings.s3_region,
        )

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self._client.head_bucket(Bucket=bucket)
        except ClientError:
            self._client.create_bucket(Bucket=bucket)

    def ensure_zones(self) -> None:
        self.ensure_bucket(self.settings.raw_bucket)
        self.ensure_bucket(self.settings.processed_bucket)

    def put_dataframe(self, bucket: str, key: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self._client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

    def get_dataframe(self, bucket: str, key: str) -> pd.DataFrame:
        obj = self._client.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    def put_json(self, bucket: str, key: str, payload: dict) -> None:
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self._client.put_object(Bucket=bucket, Key=key, Body=body)

    def get_json(self, bucket: str, key: str) -> dict:
        obj = self._client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())

    def list_keys(self, bucket: str, prefix: str) -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(o["Key"] for o in page.get("Contents", []))
        return keys
