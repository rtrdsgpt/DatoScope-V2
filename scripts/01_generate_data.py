"""
01_generate_data.py
DatoScope — Step 1: Synthetic Dataset Generation

Generates all datasets needed for clustering and regression experiments.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import (
    build_registry,
    DATASETS_DIR,
    ensure_dirs,
    gen_anisotropic_blobs,
    gen_circles,
    gen_gaussian_blobs,
    gen_highdim_regression,
    gen_linear_regression,
    gen_moons,
    gen_nonlinear_regression,
    gen_sinusoidal_regression,
    gen_spiral,
    gen_variable_density,
    save_dataset,
)


SEED = 42


def main():
    ensure_dirs()

    print("=" * 62)
    print("  DatoScope · 01_generate_data.py")
    print("=" * 62)

    print("\n── Clustering datasets ─────────────────────────────────────")
    specs = [
        ("clustering_gaussian_blobs.csv", gen_gaussian_blobs(seed=SEED)),
        ("clustering_moons.csv", gen_moons(seed=SEED)),
        ("clustering_circles.csv", gen_circles(seed=SEED)),
        ("clustering_spiral.csv", gen_spiral(seed=SEED)),
        ("clustering_anisotropic.csv", gen_anisotropic_blobs(seed=SEED)),
        ("clustering_variable_density.csv", gen_variable_density(seed=SEED)),
    ]
    for fname, df in specs:
        save_dataset(df, fname)
        missing = df.isnull().sum().sum()
        print(f"  [OK] {fname:<42} shape={df.shape}  missing={missing}")

    print("\n── Regression datasets ──────────────────────────────────────")
    reg_specs = [
        ("regression_linear.csv", gen_linear_regression(seed=SEED)),
        ("regression_nonlinear.csv", gen_nonlinear_regression(seed=SEED)),
        ("regression_sinusoidal.csv", gen_sinusoidal_regression(seed=SEED)),
    ]
    for fname, df in reg_specs:
        save_dataset(df, fname)
        missing = df.isnull().sum().sum()
        print(f"  [OK] {fname:<42} shape={df.shape}  missing={missing}")

    df_hd, df_coef = gen_highdim_regression(seed=SEED)
    save_dataset(df_hd, "regression_highdim.csv")
    save_dataset(df_coef, "regression_highdim_true_coefs.csv")
    print(f"  [OK] {'regression_highdim.csv':<42} shape={df_hd.shape}  missing={df_hd.isnull().sum().sum()}")
    print(f"  [OK] {'regression_highdim_true_coefs.csv':<42} shape={df_coef.shape}")

    for sigma in (1.0, 5.0, 20.0):
        df = gen_linear_regression(n_samples=500, noise=sigma, seed=SEED, outlier_pct=0.0)
        tag = str(sigma).replace(".", "_")
        fname = f"regression_noisy_sigma{tag}.csv"
        save_dataset(df, fname)
        print(f"  [OK] {fname:<42} shape={df.shape}  σ={sigma}")

    print("\n── Building registry ────────────────────────────────────────")
    registry = build_registry("datasets")
    reg_path = os.path.join(DATASETS_DIR, "_registry.csv")
    registry.to_csv(reg_path, index=False)
    print(f"  [OK] _registry.csv  ({len(registry)} datasets)\n")

    print(registry[["filename", "task", "n_rows", "n_cols"]].to_string(index=False))
    print("\n" + "=" * 62)
    print("  All datasets saved to:  datasets/")
    print("=" * 62)


if __name__ == "__main__":
    main()
