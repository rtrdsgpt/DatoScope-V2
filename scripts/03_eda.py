"""
03_eda.py
DatoScope — Step 3: Exploratory Data Analysis

For every cleaned dataset this script computes and prints:
  • Summary statistics       (mean, std, median, min, max, skew, kurtosis)
  • Class / label balance    (clustering datasets)
  • Feature variance ranking
  • Correlation matrix       (Pearson r) + top correlated pairs
  • Automated insights       (mirrors "Insights & Correlations" tab in HTML)

All results are also saved to outputs/_eda_report.csv.

Run:
    python 03_eda.py
    (Run 01_generate_data.py and 02_clean_data.py first.)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from itertools import combinations

from utils import load_output, list_datasets, OUTPUTS_DIR, ensure_dirs


# ════════════════════════════════════════════════════════════════════════════
# 1. SUMMARY STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def summary_statistics(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Per-feature descriptive statistics.
    Mirrors the 'Summary statistics' card in the HTML explorer's EDA tab.
    """
    rows = []
    for col in feature_cols:
        s = df[col].dropna()
        if not pd.api.types.is_numeric_dtype(s) or s.empty:
            continue
        rows.append({
            "feature"  : col,
            "count"    : int(s.count()),
            "mean"     : round(s.mean(), 4),
            "std"      : round(s.std(),  4),
            "median"   : round(s.median(), 4),
            "min"      : round(s.min(), 4),
            "max"      : round(s.max(), 4),
            "skewness" : round(float(sp_stats.skew(s)), 4),
            "kurtosis" : round(float(sp_stats.kurtosis(s)), 4),
            "q25"      : round(s.quantile(0.25), 4),
            "q75"      : round(s.quantile(0.75), 4),
        })
    if not rows:
        return pd.DataFrame(columns=["feature", "count", "mean", "std", "median", "min", "max", "skewness", "kurtosis", "q25", "q75"])
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 2. CLASS / LABEL BALANCE
# ════════════════════════════════════════════════════════════════════════════

def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count samples per class label.
    Only relevant for clustering datasets with a 'label' column.
    Mirrors the 'Class balance' bar chart in the HTML explorer.
    """
    if "label" not in df.columns:
        return pd.DataFrame(columns=["label", "count", "pct"])

    counts = df["label"].value_counts().sort_index().reset_index()
    counts.columns = ["label", "count"]
    counts["pct"]  = (counts["count"] / len(df) * 100).round(2)
    return counts


# ════════════════════════════════════════════════════════════════════════════
# 3. FEATURE VARIANCE RANKING
# ════════════════════════════════════════════════════════════════════════════

def feature_variance_ranking(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Rank features by variance (descending).
    Mirrors the 'Feature variance ranking' chart in the HTML EDA tab.
    High variance → more discriminative; near-zero → likely noise.
    """
    num_df = df[feature_cols].select_dtypes(include=np.number)
    if num_df.empty:
        return pd.DataFrame(columns=["feature", "variance", "std", "rank"])
    variances = (
        num_df
        .var()
        .reset_index()
        .rename(columns={"index": "feature", 0: "variance"})
    )
    variances.columns = ["feature", "variance"]
    variances["variance"] = variances["variance"].round(4)
    variances["std"]      = variances["variance"].apply(lambda v: round(v**0.5, 4))
    variances["rank"]     = variances["variance"].rank(ascending=False).astype(int)
    return variances.sort_values("variance", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# 4. CORRELATION MATRIX
# ════════════════════════════════════════════════════════════════════════════

def correlation_matrix(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Full Pearson correlation matrix.
    Mirrors the correlation heatmap in the HTML Insights tab.
    """
    num_df = df[feature_cols].select_dtypes(include=np.number)
    if num_df.empty:
        return pd.DataFrame()
    return num_df.corr(method="pearson").round(4)


def top_correlated_pairs(corr_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Extract the top-n most strongly correlated feature pairs (|r| desc).
    Mirrors the 'Top feature correlations' table in the HTML Insights tab.
    """
    features = corr_df.columns.tolist()
    pairs    = []
    for a, b in combinations(features, 2):
        r = corr_df.loc[a, b]
        pairs.append({"feature_a": a, "feature_b": b, "pearson_r": r, "abs_r": abs(r)})

    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r"])

    df_pairs = pd.DataFrame(pairs).sort_values("abs_r", ascending=False).head(top_n)
    return df_pairs.drop(columns="abs_r").reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# 5. AUTOMATED INSIGHTS  (mirrors HTML autoInsights div)
# ════════════════════════════════════════════════════════════════════════════

def automated_insights(
    df: pd.DataFrame,
    feature_cols: list,
    corr_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    dataset_name: str,
) -> list:
    """
    Generate a list of plain-text insight strings — same logic as the
    JavaScript autoInsights block in the HTML explorer.
    """
    insights = []

    # 1. Dataset summary
    task = "clustering" if "clustering" in dataset_name else "regression"
    insights.append(
        f"Dataset '{dataset_name}' — {task} task, "
        f"{len(df)} samples × {len(feature_cols)} features."
    )

    # 2. Most correlated pair
    if not pairs_df.empty:
        top = pairs_df.iloc[0]
        insights.append(
            f"Strongest correlation: {top.feature_a} & {top.feature_b}  "
            f"(r = {round(top.pearson_r, 3)})"
        )

    # 3. Highest variance feature
    var_rank = feature_variance_ranking(df, feature_cols)
    if not var_rank.empty:
        top_var  = var_rank.iloc[0]
        insights.append(
            f"Highest variance feature: {top_var.feature}  (σ² = {top_var.variance})"
        )

        bot_var = var_rank.iloc[-1]
        if bot_var.variance < 0.01:
            insights.append(
                f"Near-zero variance: '{bot_var.feature}' (σ² = {bot_var.variance}) "
                f"— may be a noise / constant column."
            )

    # 5. Skewed features
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]) or df[col].dropna().empty:
            continue
        skew = round(float(sp_stats.skew(df[col].dropna())), 2)
        if abs(skew) > 1.0:
            direction = "right" if skew > 0 else "left"
            insights.append(
                f"Feature '{col}' is {direction}-skewed (skewness = {skew}). "
                f"Consider log/Box-Cox transform before regression."
            )

    # 6. Class balance (clustering)
    if "label" in df.columns:
        balance = class_balance(df)
        n_classes = len(balance)
        min_pct   = balance["pct"].min()
        max_pct   = balance["pct"].max()
        insights.append(
            f"{n_classes} classes detected. "
            f"Class sizes range from {min_pct}% to {max_pct}% of data."
        )
        if max_pct / (min_pct + 1e-9) > 3:
            insights.append(
                "Class imbalance detected — consider stratified sampling or "
                "weighted clustering metrics."
            )

    # 7. Target range (regression)
    if "target" in df.columns:
        t = df["target"].dropna()
        if pd.api.types.is_numeric_dtype(t) and not t.empty:
            insights.append(
                f"Regression target range: [{round(t.min(),2)}, {round(t.max(),2)}]  "
                f"mean={round(t.mean(),2)}  std={round(t.std(),2)}"
            )

    return insights


# ════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def print_section(title: str):
    print(f"\n  ── {title} {'─'*(52 - len(title))}")


def print_eda(
    dataset_name: str,
    stats_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    var_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    insights: list,
):
    w = 62
    print(f"\n{'='*w}")
    print(f"  {dataset_name}")
    print(f"{'='*w}")

    print_section("Summary statistics")
    print(stats_df[["feature","mean","std","median","min","max","skewness"]].to_string(index=False))

    if not balance_df.empty:
        print_section("Class balance")
        print(balance_df.to_string(index=False))

    print_section("Feature variance ranking")
    print(var_df[["rank","feature","variance","std"]].to_string(index=False))

    print_section("Top correlated pairs  (Pearson r)")
    if not pairs_df.empty:
        print(pairs_df.head(6).to_string(index=False))
    else:
        print("  Only one feature — no pairs.")

    print_section("Automated insights")
    for i, ins in enumerate(insights, 1):
        print(f"  {i}. {ins}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    print("=" * 62)
    print("  DatoScope · 03_eda.py")
    print("=" * 62)

    cleaned_files = [
        f for f in os.listdir(OUTPUTS_DIR)
        if f.endswith("_cleaned.csv")
    ]

    if not cleaned_files:
        print("\n  ERROR: No cleaned datasets found in outputs/")
        print("  Run 02_clean_data.py first.\n")
        return

    all_stats    = []
    all_balance  = []
    all_variance = []
    all_pairs    = []

    for fname in sorted(cleaned_files):
        dataset_name = fname.replace("_cleaned.csv", "")
        df           = load_output(fname)

        label_cols   = [c for c in df.columns if c in ("label", "target")]
        feature_cols = [c for c in df.columns if c not in label_cols]

        # Run analyses
        stats_df   = summary_statistics(df, feature_cols)
        balance_df = class_balance(df)
        var_df     = feature_variance_ranking(df, feature_cols)
        corr_df    = correlation_matrix(df, feature_cols)
        pairs_df   = top_correlated_pairs(corr_df, top_n=10)
        insights   = automated_insights(df, feature_cols, corr_df, pairs_df, dataset_name)

        # Print
        print_eda(dataset_name, stats_df, balance_df, var_df, pairs_df, insights)

        # Collect for CSV export
        stats_df.insert(0, "dataset", dataset_name)
        all_stats.append(stats_df)

        if not balance_df.empty:
            balance_df.insert(0, "dataset", dataset_name)
            all_balance.append(balance_df)

        var_df.insert(0, "dataset", dataset_name)
        all_variance.append(var_df)

        if not pairs_df.empty:
            pairs_df.insert(0, "dataset", dataset_name)
            all_pairs.append(pairs_df)

        # Save per-dataset correlation matrix
        corr_path = os.path.join(OUTPUTS_DIR, f"{dataset_name}_corr_matrix.csv")
        corr_df.to_csv(corr_path)

    # Save consolidated EDA tables
    def _save(frames, filename):
        if frames:
            out = pd.concat(frames, ignore_index=True)
            path = os.path.join(OUTPUTS_DIR, filename)
            out.to_csv(path, index=False)
            print(f"\n  [saved] {filename}")
            return out
        return pd.DataFrame()

    print("\n" + "=" * 62)
    print("  Saving EDA outputs …")
    _save(all_stats,    "_eda_summary_stats.csv")
    _save(all_balance,  "_eda_class_balance.csv")
    _save(all_variance, "_eda_variance_ranking.csv")
    _save(all_pairs,    "_eda_top_correlations.csv")

    print("\n  Correlation matrices → outputs/*_corr_matrix.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
