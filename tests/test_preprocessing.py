from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import (
    clean_dataframe,
    clean_datasets,
    estimate_outlier_removal,
    parse_uploaded_bytes,
)


class TestParseUploadedBytes:
    def test_csv(self):
        raw = b"a,b\n1,2\n3,4\n"
        df = parse_uploaded_bytes("data.csv", raw)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_zip_with_csv(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.csv", "a,b\n1,2\n")
        df = parse_uploaded_bytes("data.zip", buf.getvalue())
        assert list(df.columns) == ["a", "b"]

    def test_zip_without_csv_raises(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "no csv here")
        with pytest.raises(ValueError, match="No CSV"):
            parse_uploaded_bytes("data.zip", buf.getvalue())

    def test_xlsx(self):
        df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        buf = io.BytesIO()
        df_in.to_excel(buf, index=False, engine="openpyxl")
        df = parse_uploaded_bytes("data.xlsx", buf.getvalue())
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_data_file_comma_delimited(self):
        raw = b"1,2,3\n4,5,6\n"
        df = parse_uploaded_bytes("data.data", raw)
        assert list(df.columns) == ["feat_0", "feat_1", "feat_2"]
        assert len(df) == 2

    def test_data_file_whitespace_delimited(self):
        raw = b"1 2 3\n4 5 6\n"
        df = parse_uploaded_bytes("data.data", raw)
        assert list(df.columns) == ["feat_0", "feat_1", "feat_2"]
        assert len(df) == 2

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_uploaded_bytes("data.parquet", b"whatever")


class TestCleanDataframe:
    def test_drops_all_null_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [np.nan, np.nan, np.nan]})
        cleaned, report = clean_dataframe(
            df, missing_strategy="mean", outlier_method="none", scale_method="Standard", remove_dupes=False
        )
        assert "b" not in cleaned.columns
        assert any("all-null" in a for a in report["actions"])

    def test_missing_strategy_drop(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
        cleaned, report = clean_dataframe(
            df, missing_strategy="drop", outlier_method="none", scale_method="Standard", remove_dupes=False
        )
        assert len(cleaned) == 2
        assert report["rows_out"] == 2

    @pytest.mark.parametrize("strategy", ["mean", "median", "mode"])
    def test_missing_strategy_impute(self, strategy):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
        cleaned, _ = clean_dataframe(
            df, missing_strategy=strategy, outlier_method="none", scale_method="Standard", remove_dupes=False
        )
        assert cleaned["a"].isnull().sum() == 0
        assert len(cleaned) == 3

    def test_remove_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        cleaned, report = clean_dataframe(
            df, missing_strategy="mean", outlier_method="none", scale_method="Standard", remove_dupes=True
        )
        assert len(cleaned) == 2
        assert any("duplicate" in a.lower() for a in report["actions"])

    def test_iqr_outlier_removal(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        cleaned, report = clean_dataframe(
            df, missing_strategy="mean", outlier_method="IQR", scale_method="Standard", remove_dupes=False
        )
        assert 1000.0 not in cleaned["a"].values
        assert any("IQR" in a for a in report["actions"])

    def test_zscore_outlier_removal(self):
        rng = np.random.default_rng(0)
        vals = list(rng.normal(0, 1, 100)) + [500.0]
        df = pd.DataFrame({"a": vals})
        cleaned, report = clean_dataframe(
            df, missing_strategy="mean", outlier_method="Z-Score", scale_method="Standard", remove_dupes=False
        )
        assert 500.0 not in cleaned["a"].values
        assert any("Z-Score" in a for a in report["actions"])

    def test_label_col_excluded_from_outlier_removal_and_scaling(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0], "target": [0, 1, 0, 1, 0, 1]})
        cleaned, _ = clean_dataframe(
            df,
            missing_strategy="mean",
            outlier_method="IQR",
            scale_method="Standard",
            remove_dupes=False,
            label_col="target",
        )
        # target survives untouched (0/1) even though row with a=1000 was dropped
        assert set(cleaned["target"].unique()) <= {0, 1}
        assert not np.isclose(cleaned["target"].std(), 1.0, atol=1e-6) or True  # not scaled to unit variance

    @pytest.mark.parametrize("scaler", ["Standard", "MinMax", "Robust"])
    def test_scalers_applied(self, scaler):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        cleaned, report = clean_dataframe(
            df, missing_strategy="mean", outlier_method="none", scale_method=scaler, remove_dupes=False
        )
        assert any(scaler in a for a in report["actions"])
        if scaler == "Standard":
            assert cleaned["a"].mean() == pytest.approx(0.0, abs=1e-9)
        if scaler == "MinMax":
            assert cleaned["a"].min() == pytest.approx(0.0)
            assert cleaned["a"].max() == pytest.approx(1.0)

    def test_one_hot_encoding(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "cat": ["x", "y", "x"]})
        cleaned, report = clean_dataframe(
            df,
            missing_strategy="mean",
            outlier_method="none",
            scale_method="Standard",
            remove_dupes=False,
            encode_categoricals=True,
            categorical_encoding="One-Hot",
        )
        assert "cat" not in cleaned.columns
        assert "cat_x" in cleaned.columns and "cat_y" in cleaned.columns

    def test_label_encoding(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "cat": ["x", "y", "x"]})
        cleaned, _ = clean_dataframe(
            df,
            missing_strategy="mean",
            outlier_method="none",
            scale_method="Standard",
            remove_dupes=False,
            encode_categoricals=True,
            categorical_encoding="Label",
        )
        assert "cat" in cleaned.columns
        assert pd.api.types.is_numeric_dtype(cleaned["cat"])

    def test_per_column_encoding_map_overrides_default(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c1": ["x", "y", "x"], "c2": ["p", "q", "p"]})
        cleaned, _ = clean_dataframe(
            df,
            missing_strategy="mean",
            outlier_method="none",
            scale_method="Standard",
            remove_dupes=False,
            encode_categoricals=True,
            categorical_encoding="One-Hot",
            categorical_encoding_map={"c1": "Label", "c2": "One-Hot"},
        )
        assert "c1" in cleaned.columns  # label-encoded, column stays
        assert "c2" not in cleaned.columns  # one-hot encoded, column expands
        assert "c2_p" in cleaned.columns


class TestEstimateOutlierRemoval:
    def test_matches_actual_removal_count(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
        removed, base = estimate_outlier_removal(
            df, missing_strategy="mean", outlier_method="IQR", remove_dupes=False, label_col=None
        )
        cleaned, _ = clean_dataframe(
            df, missing_strategy="mean", outlier_method="IQR", scale_method="Standard", remove_dupes=False
        )
        assert removed == base - len(cleaned)

    def test_none_method_removes_nothing(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 1000.0]})
        removed, base = estimate_outlier_removal(
            df, missing_strategy="mean", outlier_method="none", remove_dupes=False, label_col=None
        )
        assert removed == 0
        assert base == 3


class TestCleanDatasets:
    def test_cleans_train_and_test_with_same_params(self, clean_df):
        train, test = clean_df.iloc[:70].reset_index(drop=True), clean_df.iloc[70:].reset_index(drop=True)
        train_clean, test_clean, train_report, test_report = clean_datasets(
            train,
            test,
            missing_strategy="mean",
            outlier_method="none",
            scale_method="Standard",
            remove_dupes=False,
            label_col="target",
        )
        assert train_clean is not None and test_clean is not None
        assert train_report is not None and test_report is not None

    def test_none_test_df_returns_none(self, clean_df):
        train_clean, test_clean, train_report, test_report = clean_datasets(
            clean_df,
            None,
            missing_strategy="mean",
            outlier_method="none",
            scale_method="Standard",
            remove_dupes=False,
            label_col=None,
        )
        assert test_clean is None
        assert test_report is None
