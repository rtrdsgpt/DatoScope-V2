"""
Data-quality test layer (todo.md section 3): asserts the Great Expectations
checks in etl/validate.py actually catch bad data, feeding known-bad
fixtures through the validation stage — not just that good data passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from etl.validate import DataQualityError, validate_dataframe


@pytest.fixture
def good_df():
    return pd.DataFrame({"a": range(20), "b": [float(i) * 1.5 for i in range(20)], "label": [i % 2 for i in range(20)]})


class TestGoodDataPasses:
    def test_clean_data_passes(self, good_df):
        report = validate_dataframe(good_df, dataset_name="good")
        assert report["success"] is True
        assert report["n_failed"] == 0

    def test_required_columns_present_passes(self, good_df):
        report = validate_dataframe(good_df, dataset_name="good", required_columns=["a", "b", "label"])
        assert report["success"] is True


class TestBadDataFailsLoudly:
    def test_missing_required_column_raises(self, good_df):
        with pytest.raises(DataQualityError) as exc_info:
            validate_dataframe(good_df, dataset_name="bad", required_columns=["a", "does_not_exist"])
        report = exc_info.value.report
        assert report["success"] is False
        assert any(f["expectation"] == "expect_column_to_exist" for f in report["failed_expectations"])

    def test_excess_nulls_raises(self):
        df = pd.DataFrame({"a": [1, None, None, None, 5], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
        with pytest.raises(DataQualityError) as exc_info:
            validate_dataframe(df, dataset_name="bad", max_null_fraction=0.2)
        report = exc_info.value.report
        assert any(f["expectation"] == "expect_column_values_to_not_be_null" for f in report["failed_expectations"])

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=float)})
        with pytest.raises(DataQualityError) as exc_info:
            validate_dataframe(df, dataset_name="bad")
        report = exc_info.value.report
        assert any(f["expectation"] == "expect_table_row_count_to_be_between" for f in report["failed_expectations"])

    def test_column_bounds_violation_raises(self):
        df = pd.DataFrame({"age": [25, 30, -5, 150, 40]})  # -5 and 150 are implausible ages
        with pytest.raises(DataQualityError) as exc_info:
            validate_dataframe(df, dataset_name="bad", column_bounds={"age": {"min": 0, "max": 120}})
        report = exc_info.value.report
        assert any(f["expectation"] == "expect_column_values_to_be_between" for f in report["failed_expectations"])

    def test_raise_on_failure_false_returns_report_instead(self):
        df = pd.DataFrame({"a": [1, None, None], "b": [1.0, 2.0, 3.0]})
        report = validate_dataframe(df, dataset_name="bad", max_null_fraction=0.1, raise_on_failure=False)
        assert report["success"] is False
        assert report["n_failed"] > 0

    def test_error_message_includes_failure_count(self, good_df):
        bad_df = good_df.copy()
        bad_df["a"] = np.nan
        with pytest.raises(DataQualityError, match=r"\d+ of \d+ expectation"):
            validate_dataframe(bad_df, dataset_name="bad", max_null_fraction=0.1)
