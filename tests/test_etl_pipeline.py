"""
Full extract -> transform -> validate -> load integration test against real
docker-compose MinIO + Postgres. Run `docker compose up -d minio warehouse`
first; otherwise skips (see tests/conftest.py's minio_ready/warehouse_engine
fixtures).
"""

from __future__ import annotations

import uuid

import pandas as pd
import pytest

from etl.load import get_dataset
from etl.pipeline import run_pipeline
from etl.validate import DataQualityError

pytestmark = pytest.mark.integration


@pytest.fixture
def dataset_name():
    return f"pytest_pipeline_{uuid.uuid4().hex[:10]}"


class TestRunPipeline:
    def test_generated_source_end_to_end(self, minio_ready, warehouse_engine, dataset_name):
        df = pd.DataFrame({"a": range(50), "b": [float(i) * 2 for i in range(50)], "target": [i % 2 for i in range(50)]})
        result = run_pipeline(
            "generated",
            dataset_name=dataset_name,
            extract_kwargs={"df": df, "generator_meta": {"kind": "test"}},
            transform_kwargs={"missing_strategy": "mean", "outlier_method": "none", "scale_method": "Standard", "remove_dupes": True},
        )

        assert result["validation"]["success"] is True
        assert result["load"]["n_rows"] == 50

        stored = get_dataset(dataset_name, result["processed"]["run_id"], engine=warehouse_engine)
        assert len(stored) == 50

    def test_uploaded_source_end_to_end(self, minio_ready, warehouse_engine, dataset_name):
        raw_bytes = b"a,b\n" + b"\n".join(f"{i},{i * 2.0}".encode() for i in range(30))
        result = run_pipeline(
            "uploaded",
            dataset_name=dataset_name,
            extract_kwargs={"filename": "data.csv", "raw_bytes": raw_bytes},
        )
        assert result["load"]["n_rows"] == 30

    def test_bad_data_fails_validation_and_does_not_load(self, minio_ready, warehouse_engine, dataset_name):
        # required_columns includes a column that doesn't exist -> validation must fail
        df = pd.DataFrame({"a": range(20)})
        with pytest.raises(DataQualityError):
            run_pipeline(
                "generated",
                dataset_name=dataset_name,
                extract_kwargs={"df": df, "generator_meta": {}},
                transform_kwargs={"missing_strategy": "mean", "outlier_method": "none", "scale_method": "Standard", "remove_dupes": False},
                validate_kwargs={"required_columns": ["a", "target"]},
            )
        # nothing should have been loaded into the warehouse
        from etl.load import DatasetNotFoundError

        with pytest.raises(DatasetNotFoundError):
            get_dataset(dataset_name, engine=warehouse_engine)
