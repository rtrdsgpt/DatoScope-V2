from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etl.extract import extract_generated
from etl.transform import transform_raw


@pytest.fixture
def raw_ref(moto_store):
    df = pd.DataFrame(
        {
            "a": list(range(20)) + [1000],  # trailing outlier
            "b": [float(i) for i in range(21)],
            "target": [i % 2 for i in range(21)],
        }
    )
    df.loc[0, "a"] = np.nan
    return extract_generated(df, "transform_test", {}, store=moto_store)


class TestTransformRaw:
    def test_writes_processed_zone_with_fresh_run_id(self, moto_store, raw_ref):
        result = transform_raw(
            raw_ref["bucket"],
            raw_ref["data_key"],
            "transform_test",
            raw_ref["run_id"],
            missing_strategy="mean",
            outlier_method="IQR",
            scale_method="Standard",
            remove_dupes=True,
            store=moto_store,
        )
        assert result["run_id"] != raw_ref["run_id"]
        assert result["source_run_id"] == raw_ref["run_id"]

        stored = moto_store.get_dataframe(result["bucket"], result["data_key"])
        pd.testing.assert_frame_equal(stored, result["df"].reset_index(drop=True))
        assert 1000 not in stored["a"].values  # IQR outlier removed
        assert stored["a"].isnull().sum() == 0  # mean-imputed

    def test_repeated_transform_produces_distinct_runs(self, moto_store, raw_ref):
        r1 = transform_raw(
            raw_ref["bucket"], raw_ref["data_key"], "transform_test", raw_ref["run_id"],
            missing_strategy="mean", outlier_method="none", scale_method="Standard", remove_dupes=False,
            store=moto_store,
        )
        r2 = transform_raw(
            raw_ref["bucket"], raw_ref["data_key"], "transform_test", raw_ref["run_id"],
            missing_strategy="median", outlier_method="IQR", scale_method="MinMax", remove_dupes=False,
            store=moto_store,
        )
        assert r1["run_id"] != r2["run_id"]
        assert r1["source_run_id"] == r2["source_run_id"] == raw_ref["run_id"]
        # different cleaning params -> different row counts (r2 drops the outlier)
        assert len(r1["df"]) != len(r2["df"])

    def test_label_col_preserved_in_metadata_and_unscaled(self, moto_store, raw_ref):
        result = transform_raw(
            raw_ref["bucket"], raw_ref["data_key"], "transform_test", raw_ref["run_id"],
            missing_strategy="mean", outlier_method="none", scale_method="Standard", remove_dupes=False,
            label_col="target", store=moto_store,
        )
        metadata = moto_store.get_json(result["bucket"], result["meta_key"])
        assert metadata["label_col"] == "target"
        assert set(result["df"]["target"].unique()) <= {0, 1}

    def test_report_reflects_actions_taken(self, moto_store, raw_ref):
        result = transform_raw(
            raw_ref["bucket"], raw_ref["data_key"], "transform_test", raw_ref["run_id"],
            missing_strategy="drop", outlier_method="IQR", scale_method="Standard", remove_dupes=True,
            store=moto_store,
        )
        report = result["report"]
        assert report["rows_in"] == 21
        assert report["rows_out"] < report["rows_in"]
        assert len(report["actions"]) > 0
