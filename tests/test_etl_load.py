"""
Integration tests against the real docker-compose warehouse Postgres.
Run `docker compose up -d warehouse` first; otherwise these skip (see
tests/conftest.py's warehouse_engine fixture).
"""

from __future__ import annotations

import uuid

import pandas as pd
import pytest

from etl.load import DatasetNotFoundError, get_dataset, get_run, list_datasets, list_runs, load_to_warehouse

pytestmark = pytest.mark.integration


@pytest.fixture
def dataset_name():
    # unique per test run so parallel/repeated runs don't collide in the shared warehouse
    return f"pytest_{uuid.uuid4().hex[:10]}"


class TestLoadToWarehouse:
    def test_load_and_read_back(self, warehouse_engine, dataset_name):
        df = pd.DataFrame({"a": range(10), "b": [float(i) for i in range(10)]})
        summary = load_to_warehouse(df, dataset_name=dataset_name, run_id="run1", source="test", engine=warehouse_engine)
        assert summary["n_rows"] == 10
        assert summary["table_name"] == f"ds_{dataset_name}"

        loaded = get_dataset(dataset_name, "run1", engine=warehouse_engine)
        assert len(loaded) == 10
        assert list(loaded.columns) == ["a", "b"]  # internal _run_id/_loaded_at columns stripped

    def test_multiple_runs_coexist(self, warehouse_engine, dataset_name):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [4, 5]})
        load_to_warehouse(df1, dataset_name=dataset_name, run_id="run1", source="test", engine=warehouse_engine)
        load_to_warehouse(df2, dataset_name=dataset_name, run_id="run2", source="test", engine=warehouse_engine)

        runs = list_runs(dataset_name, engine=warehouse_engine)
        assert {r["run_id"] for r in runs} == {"run1", "run2"}

        assert len(get_dataset(dataset_name, "run1", engine=warehouse_engine)) == 3
        assert len(get_dataset(dataset_name, "run2", engine=warehouse_engine)) == 2

    def test_get_dataset_defaults_to_latest_run(self, warehouse_engine, dataset_name):
        load_to_warehouse(pd.DataFrame({"a": [1]}), dataset_name=dataset_name, run_id="run1", source="test", engine=warehouse_engine)
        load_to_warehouse(pd.DataFrame({"a": [1, 2]}), dataset_name=dataset_name, run_id="run2", source="test", engine=warehouse_engine)

        latest = get_dataset(dataset_name, engine=warehouse_engine)
        latest_run = get_run(dataset_name, engine=warehouse_engine)
        assert latest_run["run_id"] == "run2"
        assert len(latest) == 2

    def test_unknown_dataset_raises(self, warehouse_engine):
        with pytest.raises(DatasetNotFoundError):
            get_dataset("totally_unknown_dataset_xyz", engine=warehouse_engine)

    def test_unknown_run_id_raises(self, warehouse_engine, dataset_name):
        load_to_warehouse(pd.DataFrame({"a": [1]}), dataset_name=dataset_name, run_id="run1", source="test", engine=warehouse_engine)
        with pytest.raises(DatasetNotFoundError):
            get_dataset(dataset_name, run_id="does_not_exist", engine=warehouse_engine)

    def test_list_datasets_includes_this_one(self, warehouse_engine, dataset_name):
        load_to_warehouse(pd.DataFrame({"a": [1]}), dataset_name=dataset_name, run_id="run1", source="test", engine=warehouse_engine)
        names = {d["dataset_name"] for d in list_datasets(engine=warehouse_engine)}
        assert dataset_name in names
