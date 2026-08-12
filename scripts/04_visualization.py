"""
04_visualization.py
DatoScope — Step 4: Visualizations

Produces publication-ready plots for every cleaned dataset.
All chart types mirror the HTML explorer's canvas panels:

  Scatter plots        — 2-D data coloured by class / regression target
  Distribution plots   — histogram + KDE per feature
  Class balance bar    — count per label
  Variance ranking bar — σ² per feature
  Correlation heatmap  — full Pearson r matrix
  Pair-wise scatter    — feature × feature, coloured by class
  Missing-values bar   — % missing per feature (pre-cleaning, from raw data)
  Before-vs-after box  — raw vs cleaned distributions side-by-side
  Noisy comparison     — three noise-level regression curves on one canvas

Run:
    python 04_visualization.py
    (Run scripts 01 → 03 first.)

Output: outputs/plots/*.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import combinations

from utils import (
    load_dataset, load_output, list_datasets,
    OUTPUTS_DIR, PLOTS_DIR, DATASETS_DIR, ensure_dirs,
)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE  = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#BA7517", "#533AB7", "#639922"]
BG_COLOR = "#FAFAFA"
GRID_CLR = "#E8E8E8"

sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.facecolor" : BG_COLOR,
    "axes.facecolor"   : BG_COLOR,
    "axes.edgecolor"   : "#CCCCCC",
    "axes.labelsize"   : 10,
    "axes.titlesize"   : 11,
    "axes.titleweight" : "semibold",
    "grid.color"       : GRID_CLR,
    "grid.linewidth"   : 0.6,
    "xtick.labelsize"  : 8,
    "ytick.labelsize"  : 8,
    "legend.fontsize"  : 8,
    "font.family"      : "DejaVu Sans",
})

ALPHA_POINT = 0.65
POINT_SIZE  = 18


def savefig(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [plot] {name}")


def _label_cols(df):
    return [c for c in df.columns if c in ("label", "target")]

def _feature_cols(df):
    return [c for c in df.columns if c not in ("label", "target")]

def _numeric_feature_cols(df):
    feats = _feature_cols(df)
    return df[feats].select_dtypes(include=np.number).columns.tolist()

def _class_colors(labels):
    return [PALETTE[int(l) % len(PALETTE)] for l in labels]


# ════════════════════════════════════════════════════════════════════════════
# 1. SCATTER — 2-D view coloured by class / target
#    Mirrors the "Preview (first 2 features)" canvas on the Generate tab.
# ════════════════════════════════════════════════════════════════════════════

def plot_scatter(df: pd.DataFrame, dataset_name: str):
    feat = _numeric_feature_cols(df)
    if len(feat) < 2:
        return

    x, y = feat[0], feat[1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor(BG_COLOR)

    if "label" in df.columns:
        for cls, grp in df.groupby("label"):
            ax.scatter(
                grp[x], grp[y],
                color=PALETTE[int(cls) % len(PALETTE)],
                s=POINT_SIZE, alpha=ALPHA_POINT,
                label=f"class {int(cls)}",
                edgecolors="none",
            )
        ax.legend(title="class", framealpha=0.6, markerscale=1.4)

    elif "target" in df.columns:
        sc = ax.scatter(
            df[x], df["target"],
            c=df["target"], cmap="viridis",
            s=POINT_SIZE, alpha=ALPHA_POINT, edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="target")
        ax.set_ylabel("target")

    ax.set_xlabel(x)
    ax.set_ylabel(y if "label" in df.columns else "target")
    ax.set_title(f"Scatter — {dataset_name}")
    savefig(fig, f"{dataset_name}_scatter.png")


# ════════════════════════════════════════════════════════════════════════════
# 2. DISTRIBUTION — histogram + KDE per feature
#    Mirrors the "Distribution — select feature" canvas on the EDA tab.
# ════════════════════════════════════════════════════════════════════════════

def plot_distributions(df: pd.DataFrame, dataset_name: str):
    feats = _numeric_feature_cols(df)
    n     = len(feats)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes      = np.array(axes).flatten()

    for i, col in enumerate(feats):
        ax  = axes[i]
        ser = df[col].dropna()
        ax.hist(ser, bins=25, color=PALETTE[0], alpha=0.55, density=True, edgecolor="white", linewidth=0.4)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(ser)
            xs  = np.linspace(ser.min(), ser.max(), 200)
            ax.plot(xs, kde(xs), color=PALETTE[0], linewidth=1.8)
        except Exception:
            pass
        ax.axvline(ser.mean(),   color=PALETTE[2], linewidth=1.2, linestyle="--", label=f"mean={ser.mean():.2f}")
        ax.axvline(ser.median(), color=PALETTE[1], linewidth=1.2, linestyle=":",  label=f"med={ser.median():.2f}")
        ax.set_title(col)
        ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Feature distributions — {dataset_name}", fontsize=12, y=1.01)
    fig.tight_layout()
    savefig(fig, f"{dataset_name}_distributions.png")


# ════════════════════════════════════════════════════════════════════════════
# 3. CLASS BALANCE BAR
#    Mirrors the "Class balance" chart on the EDA tab.
# ════════════════════════════════════════════════════════════════════════════

def plot_class_balance(df: pd.DataFrame, dataset_name: str):
    if "label" not in df.columns:
        return

    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    bars = ax.bar(
        [f"class {int(c)}" for c in counts.index],
        counts.values,
        color=[PALETTE[int(c) % len(PALETTE)] for c in counts.index],
        edgecolor="white", linewidth=0.5, width=0.6,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylabel("Sample count")
    ax.set_title(f"Class balance — {dataset_name}")
    fig.tight_layout()
    savefig(fig, f"{dataset_name}_class_balance.png")


# ════════════════════════════════════════════════════════════════════════════
# 4. FEATURE VARIANCE RANKING BAR
#    Mirrors the "Feature variance ranking" chart on the EDA tab.
# ════════════════════════════════════════════════════════════════════════════

def plot_variance_ranking(df: pd.DataFrame, dataset_name: str):
    feats = _numeric_feature_cols(df)
    if not feats:
        return

    variances = df[feats].var().sort_values(ascending=False)
    fig, ax   = plt.subplots(figsize=(max(5, len(feats) * 0.7), 3.5))

    ax.barh(
        variances.index[::-1],
        variances.values[::-1],
        color=PALETTE[1], alpha=0.8, edgecolor="white",
    )
    ax.set_xlabel("Variance (σ²)")
    ax.set_title(f"Feature variance ranking — {dataset_name}")
    fig.tight_layout()
    savefig(fig, f"{dataset_name}_variance.png")


# ════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION HEATMAP
#    Mirrors the "Correlation matrix" in the HTML Insights tab.
# ════════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df: pd.DataFrame, dataset_name: str):
    feats = _numeric_feature_cols(df)
    if len(feats) < 2:
        return

    corr = df[feats].corr()
    size = max(4, len(feats) * 0.9)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle mask
    sns.heatmap(
        corr, ax=ax,
        mask=~mask,          # show lower triangle + diagonal
        annot=True, fmt=".2f",
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        annot_kws={"size": 8},
        square=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(f"Pearson correlation matrix — {dataset_name}")
    fig.tight_layout()
    savefig(fig, f"{dataset_name}_correlation.png")


# ════════════════════════════════════════════════════════════════════════════
# 6. PAIR-WISE SCATTER GRID
#    Mirrors the "Scatter — feature pair" selector on the Insights tab,
#    but renders all pairs at once in a grid.
# ════════════════════════════════════════════════════════════════════════════

def plot_pairplot(df: pd.DataFrame, dataset_name: str, max_features: int = 5):
    feats = _numeric_feature_cols(df)[:max_features]
    if len(feats) < 2:
        return

    hue_col = "label" if "label" in df.columns else None
    plot_df = df[feats + ([hue_col] if hue_col else [])].dropna()

    palette_map = {int(c): PALETTE[int(c) % len(PALETTE)] for c in plot_df[hue_col].unique()} if hue_col else None

    pg = sns.PairGrid(plot_df, hue=hue_col, palette=palette_map, height=2.0, aspect=1.1)
    pg.map_diag(sns.histplot, bins=15, alpha=0.55, edgecolor="none")
    pg.map_lower(sns.scatterplot, s=10, alpha=0.55, edgecolors="none")
    pg.map_upper(sns.kdeplot, levels=4, linewidths=0.8)
    if hue_col:
        pg.add_legend(title="class", markerscale=1.5, fontsize=7)
    pg.figure.suptitle(f"Pair plot — {dataset_name}", y=1.01, fontsize=11, fontweight="semibold")
    savefig(pg.figure, f"{dataset_name}_pairplot.png")


# ════════════════════════════════════════════════════════════════════════════
# 7. MISSING VALUES BAR  (raw dataset, before cleaning)
#    Mirrors the "Missing values by feature" canvas on the Cleaning tab.
# ════════════════════════════════════════════════════════════════════════════

def plot_missing_values(df_raw: pd.DataFrame, dataset_name: str):
    feats   = _feature_cols(df_raw)
    missing = (df_raw[feats].isna().sum() / len(df_raw) * 100).sort_values(ascending=False)

    if missing.sum() == 0:
        return  # Nothing to show

    fig, ax = plt.subplots(figsize=(max(5, len(feats) * 0.7), 3.5))
    colors  = [PALETTE[2] if v > 10 else PALETTE[4] if v > 0 else PALETTE[1] for v in missing.values]

    ax.bar(missing.index, missing.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(5, color=PALETTE[4], linewidth=1, linestyle="--", label="5 % threshold")
    ax.set_ylabel("Missing (%)")
    ax.set_title(f"Missing values per feature — {dataset_name} (raw)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, f"{dataset_name}_missing.png")


# ════════════════════════════════════════════════════════════════════════════
# 8. BEFORE vs AFTER BOX PLOT  (mirrors "Before vs After" on Cleaning tab)
# ════════════════════════════════════════════════════════════════════════════

def plot_before_after(df_raw: pd.DataFrame, df_clean: pd.DataFrame, dataset_name: str):
    feats = _numeric_feature_cols(df_raw)
    if not feats:
        return

    show = feats[:4]   # up to 4 features side-by-side
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 4), sharey=False)
    if len(show) == 1:
        axes = [axes]

    for ax, col in zip(axes, show):
        raw_vals   = df_raw[col].dropna()
        clean_vals = df_clean[col].dropna() if col in df_clean.columns else pd.Series([], dtype=float)

        ax.boxplot(
            [raw_vals, clean_vals],
            labels=["raw", "cleaned"],
            patch_artist=True,
            boxprops=dict(facecolor=PALETTE[0], alpha=0.5),
            medianprops=dict(color=PALETTE[2], linewidth=2),
            whiskerprops=dict(linestyle="--"),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("value")

    fig.suptitle(f"Before vs After cleaning — {dataset_name}", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    savefig(fig, f"{dataset_name}_before_after.png")


# ════════════════════════════════════════════════════════════════════════════
# 9. NOISY REGRESSION COMPARISON  (special multi-panel chart)
#    Overlays three noise-level variants on one figure.
# ════════════════════════════════════════════════════════════════════════════

def plot_noisy_comparison():
    """Load all three noisy regression CSVs and overlay them."""
    noise_files = {
        "σ = 1.0" : "regression_noisy_sigma1_0_cleaned.csv",
        "σ = 5.0" : "regression_noisy_sigma5_0_cleaned.csv",
        "σ = 20.0": "regression_noisy_sigma20_0_cleaned.csv",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ax, (label, fname) in zip(axes, noise_files.items()):
        try:
            df = load_output(fname)
        except FileNotFoundError:
            ax.set_visible(False)
            continue

        color = PALETTE[list(noise_files).index(label)]
        ax.scatter(df["x1"], df["target"], s=8, alpha=0.4, color=color, edgecolors="none")

        # Fit and overlay a linear trendline for visual reference
        m, b = np.polyfit(df["x1"].dropna(), df["target"].dropna(), 1)
        xs   = np.linspace(df["x1"].min(), df["x1"].max(), 200)
        ax.plot(xs, m*xs + b, color="black", linewidth=1.5, linestyle="--", label=f"y={m:.1f}x+{b:.1f}")

        ax.set_title(f"Noisy regression ({label})")
        ax.set_xlabel("x1")
        ax.set_ylabel("target")
        ax.legend(fontsize=7)

    fig.suptitle("Effect of noise on regression signal", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    savefig(fig, "regression_noisy_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
# 10. HIGH-DIM REGRESSION — TRUE vs ZERO COEFFICIENT PROFILE
# ════════════════════════════════════════════════════════════════════════════

def plot_highdim_coef_profile():
    coef_path = os.path.join(DATASETS_DIR, "regression_highdim_true_coefs.csv")
    if not os.path.exists(coef_path):
        return

    coef_df = pd.read_csv(coef_path)
    fig, ax = plt.subplots(figsize=(10, 3.5))

    colors = [PALETTE[2] if abs(c) > 0 else PALETTE[1] for c in coef_df["true_coefficient"]]
    ax.bar(coef_df["feature"], coef_df["true_coefficient"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor=PALETTE[2], label="informative feature"),
        Patch(facecolor=PALETTE[1], label="noise feature (coef = 0)"),
    ]
    ax.legend(handles=legend_els, fontsize=8)
    ax.set_xlabel("feature")
    ax.set_ylabel("true coefficient")
    ax.set_title("High-dim regression — true coefficient profile")
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.tight_layout()
    savefig(fig, "regression_highdim_coef_profile.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    print("=" * 62)
    print("  DatoScope · 04_visualization.py")
    print("=" * 62)

    cleaned_files = sorted(
        f for f in os.listdir(OUTPUTS_DIR)
        if f.endswith("_cleaned.csv")
    )

    if not cleaned_files:
        print("\n  ERROR: No cleaned datasets in outputs/")
        print("  Run 02_clean_data.py first.\n")
        return

    print(f"\n  Generating plots for {len(cleaned_files)} datasets …\n")

    for fname in cleaned_files:
        dataset_name = fname.replace("_cleaned.csv", "")
        df_clean     = load_output(fname)

        # Try to load the matching raw dataset
        raw_fname = fname.replace("_cleaned.csv", ".csv")
        try:
            df_raw = load_dataset(raw_fname)
        except FileNotFoundError:
            df_raw = df_clean   # fall back to cleaned if raw unavailable

        print(f"  ── {dataset_name}")

        plot_scatter(df_clean,           dataset_name)
        plot_distributions(df_clean,     dataset_name)
        plot_class_balance(df_clean,     dataset_name)
        plot_variance_ranking(df_clean,  dataset_name)
        plot_correlation_heatmap(df_clean, dataset_name)
        plot_pairplot(df_clean,          dataset_name)
        plot_missing_values(df_raw,      dataset_name)
        plot_before_after(df_raw, df_clean, dataset_name)

    # Special multi-dataset plots
    print("\n  ── Special charts")
    plot_noisy_comparison()
    plot_highdim_coef_profile()

    total = len(os.listdir(PLOTS_DIR))
    print(f"\n{'='*62}")
    print(f"  {total} plots saved → outputs/plots/")
    print("=" * 62)


if __name__ == "__main__":
    main()
