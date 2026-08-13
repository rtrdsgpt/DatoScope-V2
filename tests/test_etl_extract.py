from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from etl.extract import RawRunNotFoundError, extract_generated, extract_kaggle, extract_uploaded, find_raw_run


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": range(10), "b": [float(i) * 2 for i in range(10)]})


class TestExtractGenerated:
    def test_writes_raw_zone_and_metadata(self, moto_store, sample_df):
        ref = extract_generated(sample_df, "my_dataset", {"kind": "linear"}, store=moto_store)
        assert ref["metadata"]["dataset_name"] == "my_dataset"
        assert ref["metadata"]["source"] == "generated"
        assert ref["metadata"]["n_rows"] == 10
        assert ref["metadata"]["generator_meta"] == {"kind": "linear"}

        roundtrip = moto_store.get_dataframe(ref["bucket"], ref["data_key"])
        pd.testing.assert_frame_equal(roundtrip, sample_df)

    def test_each_call_gets_a_fresh_run_id(self, moto_store, sample_df):
        ref1 = extract_generated(sample_df, "ds", {}, store=moto_store)
        ref2 = extract_generated(sample_df, "ds", {}, store=moto_store)
        assert ref1["run_id"] != ref2["run_id"]


class TestExtractUploaded:
    def test_parses_and_writes_raw_zone(self, moto_store):
        raw_bytes = b"a,b\n1,2\n3,4\n"
        ref = extract_uploaded("data.csv", raw_bytes, "uploaded_ds", store=moto_store)
        assert ref["metadata"]["source"] == "uploaded"
        assert ref["metadata"]["original_filename"] == "data.csv"
        df = moto_store.get_dataframe(ref["bucket"], ref["data_key"])
        assert list(df.columns) == ["a", "b"]

    def test_bad_file_raises_before_writing(self, moto_store):
        with pytest.raises(ValueError):
            extract_uploaded("data.parquet", b"junk", "bad_ds", store=moto_store)
        with pytest.raises(RawRunNotFoundError):
            find_raw_run("bad_ds", store=moto_store)


class TestExtractKaggle:
    def test_uses_mocked_download_and_lands_raw(self, moto_store, tmp_path, sample_df):
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)

        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            ref = extract_kaggle("someuser/somedataset", "kaggle_ds", store=moto_store)

        assert ref["metadata"]["source"] == "kaggle"
        assert ref["metadata"]["kaggle_handle"] == "someuser/somedataset"
        df = moto_store.get_dataframe(ref["bucket"], ref["data_key"])
        assert len(df) == len(sample_df)

    def test_no_csv_files_raises(self, moto_store, tmp_path):
        with patch("kagglehub.dataset_download", return_value=str(tmp_path)):
            with pytest.raises(ValueError, match="No CSV files"):
                extract_kaggle("someuser/somedataset", "kaggle_ds", store=moto_store)


class TestFindRawRun:
    def test_finds_latest_across_sources(self, moto_store, sample_df):
        extract_generated(sample_df, "shared_name", {}, store=moto_store)
        ref2 = extract_uploaded("f.csv", b"a,b\n1,2\n", "shared_name", store=moto_store)

        found = find_raw_run("shared_name", store=moto_store)
        assert found["run_id"] == ref2["run_id"]  # uploaded happened after generated

    def test_specific_run_id_lookup(self, moto_store, sample_df):
        ref1 = extract_generated(sample_df, "ds", {}, store=moto_store)
        extract_generated(sample_df, "ds", {}, store=moto_store)

        found = find_raw_run("ds", run_id=ref1["run_id"], store=moto_store)
        assert found["run_id"] == ref1["run_id"]

    def test_missing_dataset_raises(self, moto_store):
        with pytest.raises(RawRunNotFoundError):
            find_raw_run("nonexistent", store=moto_store)
