"""
02_clean_data.py
DatoScope — Step 2: Data Cleaning & Preprocessing

For every raw dataset produced by 01_generate_data.py this script:
  1. Prints a Data Quality Report  (mirrors the "Cleaning & Preprocessing"
     tab in the HTML explorer — missing %, outlier count, mean, std)
  2. Handles missing values        (mean / median / mode fill or row drop)
  3. Removes outliers              (IQR method or Z-score  |z| > 3)
  4. Removes duplicate rows
  5. Scales features               (Standard, Min-Max, or Robust)
  6. Saves cleaned CSVs to outputs/

Run:
    python 02_clean_data.py

Output: outputs/<name>_cleaned.csv  +  outputs/_cleaning_report.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

from utils import (
    load_dataset, save_output, list_datasets,
    OUTPUTS_DIR, ensure_dirs,
)

# ── Cleaning configuration ───────────────────────────────────────────────────
# Change these to experiment — mirrors the dropdowns in the HTML explorer.
CONFIG = {
    "missing_strategy" : "mean",       # "mean" | "median" | "mode" | "drop"
    "outlier_method"   : "iqr",        # "iqr"  | "zscore"
    "scale_method"     : "standard",   # "standard" | "minmax" | "robust"
    "handle_missing"   : True,
    "handle_outliers"  : True,
    "handle_duplicates": True,
    "scale_features"   : True,
}


# ════════════════════════════════════════════════════════════════════════════
# QUALITY REPORT
# ════════════════════════════════════════════════════════════════════════════

def quality_report(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Produce a per-feature quality table.
    Columns: feature, n_missing, missing_pct, n_outliers, mean, std, min, max
    Mirrors the 'Data Quality Report' card in the HTML explorer.
    """
    rows = []
    for col in feature_cols:
        series  = df[col].dropna()
        n_miss  = df[col].isna().sum()
        miss_pct = round(n_miss / len(df) * 100, 2)

        is_num = pd.api.types.is_numeric_dtype(series)

        mn, sd  = (series.mean(), series.std()) if is_num else (np.nan, np.nan)
        z_outs  = int((np.abs(stats.zscore(series)) > 3).sum()) if is_num and sd > 0 else 0

        if is_num and not series.empty:
            q1, q3  = series.quantile(0.25), series.quantile(0.75)
            iqr     = q3 - q1
            iqr_outs = int(((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum())
        else:
            iqr_outs = 0

        rows.append({
            "feature"     : col,
            "n_missing"   : int(n_miss),
            "missing_pct" : miss_pct,
            "outliers_zscore": z_outs,
            "outliers_iqr"   : iqr_outs,
            "mean"        : round(mn, 4) if pd.notna(mn) else None,
            "std"         : round(sd, 4) if pd.notna(sd) else None,
            "min"         : round(series.min(), 4) if is_num and not series.empty else None,
            "max"         : round(series.max(), 4) if is_num and not series.empty else None,
        })
    return pd.DataFrame(rows)


def print_quality_report(report: pd.DataFrame, dataset_name: str):
    """Pretty-print quality report to console."""
    total_miss   = report["n_missing"].sum()
    total_iqr    = report["outliers_iqr"].sum()

    flag_miss  = "⚠" if total_miss  > 0 else "✓"
    flag_out   = "⚠" if total_iqr   > 0 else "✓"

    print(f"\n  ┌─ Quality report: {dataset_name}")
    print(f"  │  {flag_miss} Total missing values : {total_miss}")
    print(f"  │  {flag_out} IQR outliers (total) : {total_iqr}")
    print(f"  └─ Per-feature breakdown:")
    print(report[["feature","missing_pct","outliers_iqr","mean","std"]].to_string(index=False, col_space=14))


# ════════════════════════════════════════════════════════════════════════════
# STEP-BY-STEP CLEANING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def handle_missing(df: pd.DataFrame, feature_cols: list, strategy: str) -> pd.DataFrame:
    """
    strategy:
      "mean"   → fill with column mean
      "median" → fill with column median
      "mode"   → fill with column mode (most frequent value)
      "drop"   → drop rows that have any missing value in feature_cols
    """
    df = df.copy()
    if strategy == "drop":
        before = len(df)
        df = df.dropna(subset=feature_cols)
        print(f"    missing→drop : removed {before - len(df)} rows")
        return df

    fill_map = {
        "mean"  : lambda s: s.mean(),
        "median": lambda s: s.median(),
        "mode"  : lambda s: s.mode()[0] if not s.mode().empty else s.mean(),
    }
    fn = fill_map[strategy]
    for col in feature_cols:
        n = df[col].isna().sum()
        if n > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = fn(df[col].dropna())
                df[col]  = df[col].fillna(round(fill_val, 4))
            else:
                fill_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col]  = df[col].fillna(fill_val)
            print(f"    missing→{strategy:<6} : {col}  ({n} values filled with {fill_val})")
    return df


def remove_outliers(df: pd.DataFrame, feature_cols: list, method: str) -> pd.DataFrame:
    """
    method:
      "iqr"    → remove rows where any feature is outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR]
      "zscore" → remove rows where |z| > 3 for any feature
    Mirrors the outlier removal in the HTML explorer's applyPreprocessing().
    """
    before = len(df)
    mask   = pd.Series([True] * len(df), index=df.index)

    for col in feature_cols:
        series = df[col].dropna()
        if not pd.api.types.is_numeric_dtype(series) or series.empty:
            continue
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            mask   = mask & ((df[col].isna()) | ((df[col] >= lo) & (df[col] <= hi)))
        elif method == "zscore":
            z    = (df[col] - series.mean()) / (series.std() + 1e-9)
            mask = mask & ((df[col].isna()) | (z.abs() <= 3))

    df = df[mask]
    print(f"    outliers→{method:<6} : removed {before - len(df)} rows")
    return df


def remove_duplicates(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    before = len(df)
    df     = df.drop_duplicates(subset=feature_cols)
    print(f"    duplicates     : removed {before - len(df)} rows")
    return df


def scale_features(df: pd.DataFrame, feature_cols: list, method: str) -> pd.DataFrame:
    """
    method:
      "standard" → z-score  (mean=0, std=1)
      "minmax"   → scale to [0, 1]
      "robust"   → subtract median, divide by IQR  (robust to outliers)
    Mirrors the scaler dropdown in the HTML explorer.
    """
    df = df.copy()
    for col in feature_cols:
        series = df[col].dropna()
        if not pd.api.types.is_numeric_dtype(series) or series.empty:
            continue
        if method == "standard":
            mn, sd = series.mean(), series.std()
            df[col] = ((df[col] - mn) / sd).round(6) if sd > 0 else 0.0
        elif method == "minmax":
            lo, hi = series.min(), series.max()
            df[col] = ((df[col] - lo) / (hi - lo)).round(6) if hi > lo else 0.0
        elif method == "robust":
            med = series.median()
            iqr = series.quantile(0.75) - series.quantile(0.25)
            df[col] = ((df[col] - med) / iqr).round(6) if iqr > 0 else 0.0
    print(f"    scaling→{method:<8} : applied to numeric features")
    return df


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE — run all steps on a single DataFrame
# ════════════════════════════════════════════════════════════════════════════

def clean_pipeline(df: pd.DataFrame, feature_cols: list, cfg: dict):
    """
    Apply cleaning steps in the same order as the HTML explorer:
    missing → outliers → duplicates → scaling
    """
    n_before = len(df)
    print(f"    rows before : {n_before}")

    if cfg["handle_missing"]:
        df = handle_missing(df, feature_cols, cfg["missing_strategy"])

    if cfg["handle_outliers"]:
        df = remove_outliers(df, feature_cols, cfg["outlier_method"])

    if cfg["handle_duplicates"]:
        df = remove_duplicates(df, feature_cols)

    if cfg["scale_features"]:
        df = scale_features(df, feature_cols, cfg["scale_method"])

    print(f"    rows after  : {len(df)}  (removed {n_before - len(df)} total)")
    return df.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    print("=" * 62)
    print("  DatoScope · 02_clean_data.py")
    print("=" * 62)
    print(f"\n  Config: {CONFIG}\n")

    all_reports = []
    datasets    = list_datasets("datasets")

    # Skip the true-coefficients reference file
    datasets = [f for f in datasets if "true_coefs" not in f]

    for fname in datasets:
        df   = load_dataset(fname)
        name = fname.replace(".csv", "")

        # Determine feature columns (everything except label / target)
        label_cols   = [c for c in df.columns if c in ("label", "target")]
        feature_cols = [c for c in df.columns if c not in label_cols]

        print(f"\n{'─'*62}")
        print(f"  Dataset : {fname}  shape={df.shape}")

        # Quality report
        report = quality_report(df, feature_cols)
        report.insert(0, "dataset", name)
        all_reports.append(report)
        print_quality_report(report, fname)

        # Clean
        print(f"\n  Cleaning steps:")
        df_clean = clean_pipeline(df, feature_cols, CONFIG)

        # Save
        out_fname = f"{name}_cleaned.csv"
        path = save_output(df_clean, out_fname)
        print(f"\n  → Saved: outputs/{out_fname}")

    # Consolidated quality report
    full_report = pd.concat(all_reports, ignore_index=True)
    rpt_path    = os.path.join(OUTPUTS_DIR, "_cleaning_report.csv")
    full_report.to_csv(rpt_path, index=False)

    print("\n" + "=" * 62)
    print(f"  Cleaning report  → outputs/_cleaning_report.csv")
    print(f"  Cleaned datasets → outputs/*_cleaned.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
