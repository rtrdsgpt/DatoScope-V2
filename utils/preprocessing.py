"""
Reusable data loading and preprocessing helpers for DatoScope.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def parse_uploaded_bytes(filename: str, raw: bytes) -> pd.DataFrame:
    """
    Parse raw file bytes into a DataFrame. Framework-agnostic (raises
    ValueError on bad input) so it can be reused by the ETL extract stage,
    not just the Streamlit uploader.
    """
    name = filename.lower()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), low_memory=False)

    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise ValueError("No CSV found inside the ZIP archive.")
            return pd.read_csv(io.BytesIO(zf.read(csvs[0])), low_memory=False)

    if name.endswith((".xls", ".xlsx")):
        engine = "xlrd" if name.endswith(".xls") else "openpyxl"
        df = pd.read_excel(io.BytesIO(raw), engine=engine)
        if all(df.dtypes == object):
            df2 = pd.read_excel(io.BytesIO(raw), engine=engine, header=1)
            if len(df2.select_dtypes("number").columns) > 0:
                df = df2
        return df

    if name.endswith(".data"):
        try:
            df = pd.read_csv(io.BytesIO(raw), header=None)
            if df.shape[1] == 1:
                raise ValueError
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), header=None, sep=r"\s+", engine="python")
        df.columns = [f"feat_{c}" for c in df.columns]
        return df

    raise ValueError(f"Unsupported file type: {filename}")


def load_uploaded_file(uploaded) -> pd.DataFrame:
    import streamlit as st

    try:
        return parse_uploaded_bytes(uploaded.name, uploaded.read())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()


def clean_dataframe(
    df: pd.DataFrame,
    missing_strategy: str,
    outlier_method: str,
    scale_method: str,
    remove_dupes: bool,
    label_col: str | None = None,
    encode_categoricals: bool = False,
    categorical_encoding: str = "One-Hot",
    categorical_encoding_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    report = {"rows_in": len(df), "cols_in": len(df.columns), "actions": []}
    df = df.copy()

    all_null = df.columns[df.isnull().all()].tolist()
    if all_null:
        df.drop(columns=all_null, inplace=True)
        report["actions"].append(f"Dropped {len(all_null)} all-null column(s)")

    if remove_dupes:
        n = df.duplicated().sum()
        if n:
            df.drop_duplicates(inplace=True)
            report["actions"].append(f"Removed {n} duplicate row(s)")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if label_col and label_col in num_cols:
        num_cols.remove(label_col)
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if missing_strategy == "drop":
        before = len(df)
        subset_cols = num_cols if num_cols else df.columns.tolist()
        df.dropna(subset=subset_cols, inplace=True)
        report["actions"].append(f"Dropped {before - len(df)} row(s) with missing values")
    else:
        num_strategy = {"mean": "mean", "median": "median", "mode": "most_frequent"}[missing_strategy]
        if num_cols:
            imp = SimpleImputer(strategy=num_strategy)
            df[num_cols] = imp.fit_transform(df[num_cols])
        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = cat_imp.fit_transform(df[cat_cols])
        report["actions"].append(f"Filled missing values with '{missing_strategy}'")

    feat_cols = [c for c in num_cols if c != label_col]
    if outlier_method == "IQR" and feat_cols:
        before = len(df)
        mask = pd.Series(True, index=df.index)
        for col in feat_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            mask &= df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        df = df[mask]
        report["actions"].append(f"IQR: removed {before - len(df)} outlier row(s)")
    elif outlier_method == "Z-Score" and feat_cols:
        before = len(df)
        z = np.abs(stats.zscore(df[feat_cols].fillna(0)))
        if np.ndim(z) == 1:
            z = np.array([z]).T
        df = df[(z < 3).all(axis=1)]
        report["actions"].append(f"Z-Score: removed {before - len(df)} outlier row(s)")

    scaler = {
        "Standard": StandardScaler(),
        "MinMax": MinMaxScaler(),
        "Robust": RobustScaler(),
    }.get(scale_method)
    if scaler and feat_cols:
        df[feat_cols] = scaler.fit_transform(df[feat_cols])
        report["actions"].append(f"Scaled with {scale_method}Scaler")

    if encode_categoricals:
        encode_cols = [c for c in cat_cols if c != label_col]
        if encode_cols:
            encoding_map = categorical_encoding_map or {}
            one_hot_cols = [col for col in encode_cols if encoding_map.get(col, categorical_encoding) == "One-Hot"]
            label_cols = [col for col in encode_cols if encoding_map.get(col, categorical_encoding) == "Label"]

            if label_cols:
                for col in label_cols:
                    df[col] = pd.factorize(df[col])[0]
                report["actions"].append(f"Label-encoded {len(label_cols)} categorical column(s)")

            if one_hot_cols:
                before_cols = df.shape[1]
                df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False)
                added_cols = df.shape[1] - before_cols
                report["actions"].append(f"One-hot encoded {len(one_hot_cols)} categorical column(s) into {added_cols} feature column(s)")

    report.update({"rows_out": len(df), "cols_out": len(df.columns)})
    return df, report


def estimate_outlier_removal(
    df: pd.DataFrame,
    *,
    missing_strategy: str,
    outlier_method: str,
    remove_dupes: bool,
    label_col: str | None = None,
) -> tuple[int, int]:
    working_df = df.copy()

    all_null = working_df.columns[working_df.isnull().all()].tolist()
    if all_null:
        working_df = working_df.drop(columns=all_null)

    if remove_dupes:
        working_df = working_df.drop_duplicates()

    num_cols = working_df.select_dtypes(include="number").columns.tolist()
    if label_col and label_col in num_cols:
        num_cols.remove(label_col)
    cat_cols = working_df.select_dtypes(exclude="number").columns.tolist()

    if missing_strategy == "drop":
        subset_cols = num_cols if num_cols else working_df.columns.tolist()
        working_df = working_df.dropna(subset=subset_cols)
    else:
        num_strategy = {"mean": "mean", "median": "median", "mode": "most_frequent"}[missing_strategy]
        if num_cols:
            imp = SimpleImputer(strategy=num_strategy)
            working_df[num_cols] = imp.fit_transform(working_df[num_cols])
        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            working_df[cat_cols] = cat_imp.fit_transform(working_df[cat_cols])

    feat_cols = [c for c in num_cols if c != label_col]
    if outlier_method == "none" or not feat_cols:
        return 0, len(working_df)

    if outlier_method == "IQR":
        mask = pd.Series(True, index=working_df.index)
        for col in feat_cols:
            q1, q3 = working_df[col].quantile(0.25), working_df[col].quantile(0.75)
            iqr = q3 - q1
            mask &= working_df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        removed = int((~mask).sum())
        return removed, len(working_df)

    if outlier_method == "Z-Score":
        z = np.abs(stats.zscore(working_df[feat_cols].fillna(0)))
        if np.ndim(z) == 1:
            z = np.array([z]).T
        mask = (z < 3).all(axis=1)
        removed = int((~mask).sum())
        return removed, len(working_df)

    return 0, len(working_df)


def clean_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    *,
    missing_strategy: str,
    outlier_method: str,
    scale_method: str,
    remove_dupes: bool,
    label_col: str | None,
    encode_categoricals: bool = False,
    categorical_encoding: str = "One-Hot",
    categorical_encoding_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict, dict | None]:
    train_clean, train_report = clean_dataframe(
        train_df,
        missing_strategy=missing_strategy,
        outlier_method=outlier_method,
        scale_method=scale_method,
        remove_dupes=remove_dupes,
        label_col=label_col,
        encode_categoricals=encode_categoricals,
        categorical_encoding=categorical_encoding,
        categorical_encoding_map=categorical_encoding_map,
    )
    test_clean = None
    test_report = None
    if test_df is not None:
        test_clean, test_report = clean_dataframe(
            test_df,
            missing_strategy=missing_strategy,
            outlier_method=outlier_method,
            scale_method=scale_method,
            remove_dupes=remove_dupes,
            label_col=label_col if label_col in test_df.columns else None,
            encode_categoricals=encode_categoricals,
            categorical_encoding=categorical_encoding,
            categorical_encoding_map=categorical_encoding_map,
        )
    return train_clean, test_clean, train_report, test_report
