"""
Reusable synthetic dataset generators for DatoScope.

Both the Streamlit app and scripts/01_generate_data.py import from here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons, make_regression


CLUSTERING_DATASETS = (
    "Gaussian Blobs",
    "Two Moons",
    "Concentric Circles",
    "Spiral",
    "Anisotropic Blobs",
    "Variable Density Blobs",
)

REGRESSION_DATASETS = (
    "Linear",
    "Nonlinear",
    "Sinusoidal",
    "High-Dimensional",
)

CLASSIFICATION_DATASETS = (
    "Linearly Separable",
    "Overlapping Classes",
    "Imbalanced Classes",
)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in ("label", "target", "class")]


def _expand_features(
    df: pd.DataFrame,
    total_features: int,
    *,
    seed: int,
) -> pd.DataFrame:
    """Expand a generated dataset to the requested number of feature columns."""
    df = df.copy()
    feature_cols = _feature_columns(df)
    current = len(feature_cols)
    if total_features <= current:
        return df

    rng = np.random.default_rng(seed)
    base = df[feature_cols].to_numpy(dtype=float)

    for idx in range(current + 1, total_features + 1):
        weights = rng.normal(0, 1, size=base.shape[1])
        projected = base @ weights
        noise = rng.normal(0, 0.15 + (idx / max(total_features, 1)) * 0.05, size=len(df))
        df[f"x{idx}"] = projected + noise

    return df


def _inject_missing(df: pd.DataFrame, pct: float, feature_cols: list[str], rng: np.random.Generator) -> None:
    for col in feature_cols:
        mask = rng.random(len(df)) < pct
        df.loc[mask, col] = np.nan


def _inject_outliers(df: pd.DataFrame, pct: float, feature_cols: list[str], rng: np.random.Generator) -> None:
    n_out = max(1, int(len(df) * pct))
    idx = rng.choice(len(df), n_out, replace=False)
    for i in idx:
        col = feature_cols[rng.integers(0, len(feature_cols))]
        df.loc[i, col] = float(rng.normal(0, 1) * 15)


def gen_gaussian_blobs(
    n_samples: int = 500,
    n_clusters: int = 3,
    cluster_std: float = 1.0,
    *,
    seed: int = 42,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    _inject_missing(df, missing_pct, ["x1", "x2"], rng)
    _inject_outliers(df, outlier_pct, ["x1", "x2"], rng)
    return df


def gen_moons(
    n_samples: int = 500,
    noise: float = 0.10,
    *,
    seed: int = 42,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    _inject_missing(df, missing_pct, ["x1", "x2"], rng)
    _inject_outliers(df, outlier_pct, ["x1", "x2"], rng)
    return df


def gen_circles(
    n_samples: int = 500,
    noise: float = 0.05,
    *,
    seed: int = 42,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=seed)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    _inject_missing(df, missing_pct, ["x1", "x2"], rng)
    _inject_outliers(df, outlier_pct, ["x1", "x2"], rng)
    return df


def gen_spiral(
    n_samples: int = 500,
    n_arms: int = 3,
    n_turns: float = 1.5,
    noise: float = 0.15,
    *,
    seed: int = 42,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pts_per_arm = max(1, n_samples // max(n_arms, 1))
    Xs, ys = [], []
    for arm in range(n_arms):
        theta = np.linspace(0, n_turns * 2 * np.pi, pts_per_arm)
        r = theta / max(n_turns * 2 * np.pi, 1e-9)
        offset = (2 * np.pi / n_arms) * arm
        x1 = r * np.cos(theta + offset) + rng.normal(0, noise, pts_per_arm)
        x2 = r * np.sin(theta + offset) + rng.normal(0, noise, pts_per_arm)
        Xs.append(np.column_stack([x1, x2]))
        ys.extend([arm] * pts_per_arm)

    X = np.vstack(Xs)[:n_samples]
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = ys[: len(df)]
    _inject_missing(df, missing_pct, ["x1", "x2"], rng)
    _inject_outliers(df, outlier_pct, ["x1", "x2"], rng)
    return df


def gen_anisotropic_blobs(
    n_samples: int = 500,
    *,
    seed: int = 42,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=seed)
    transform = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X = X @ transform
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    _inject_missing(df, missing_pct, ["x1", "x2"], rng)
    _inject_outliers(df, outlier_pct, ["x1", "x2"], rng)
    return df


def gen_variable_density(
    n_samples: int = 500,
    *,
    seed: int = 42,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n1, n2 = n_samples // 4, n_samples // 3
    n3 = n_samples - n1 - n2
    c1 = rng.normal([0, 0], 0.3, (n1, 2))
    c2 = rng.normal([5, 5], 1.5, (n2, 2))
    c3 = rng.normal([-4, 4], 0.8, (n3, 2))
    X = np.vstack([c1, c2, c3])
    y = np.array([0] * n1 + [1] * n2 + [2] * n3)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    _inject_missing(df, missing_pct, ["x1", "x2"], rng)
    _inject_outliers(df, outlier_pct, ["x1", "x2"], rng)
    return df


def gen_linear_regression(
    n_samples: int = 500,
    noise: float = 10.0,
    *,
    seed: int = 42,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=seed)
    df = pd.DataFrame({"x1": X[:, 0], "target": y})
    _inject_missing(df, missing_pct, ["x1"], rng)
    _inject_outliers(df, outlier_pct, ["x1"], rng)
    return df


def gen_nonlinear_regression(
    n_samples: int = 500,
    noise: float = 5.0,
    *,
    seed: int = 42,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, n_samples)
    y = 2 * x**3 - x**2 + 3 * x - 5 + rng.normal(0, noise, n_samples)
    df = pd.DataFrame({"x1": x, "target": y})
    _inject_missing(df, missing_pct, ["x1"], rng)
    _inject_outliers(df, outlier_pct, ["x1"], rng)
    return df


def gen_sinusoidal_regression(
    n_samples: int = 500,
    noise: float = 0.30,
    *,
    seed: int = 42,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, n_samples)
    y = np.sin(x) + rng.normal(0, noise, n_samples)
    df = pd.DataFrame({"x1": x, "target": y})
    _inject_missing(df, missing_pct, ["x1"], rng)
    _inject_outliers(df, outlier_pct, ["x1"], rng)
    return df


def gen_highdim_regression(
    n_samples: int = 500,
    n_features: int = 20,
    n_informative: int = 5,
    noise: float = 10.0,
    *,
    seed: int = 42,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        coef=True,
        random_state=seed,
    )
    cols = [f"x{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    coef_df = pd.DataFrame({"feature": cols, "true_coefficient": coef})
    _inject_missing(df, missing_pct, cols, rng)
    _inject_outliers(df, outlier_pct, cols, rng)
    return df, coef_df


def gen_classification(
    n_samples: int = 500,
    n_features: int = 6,
    n_informative: int = 4,
    *,
    seed: int = 42,
    class_sep: float = 1.5,
    weights: tuple[float, float] | None = None,
    flip_y: float = 0.02,
    missing_pct: float = 0.05,
    outlier_pct: float = 0.03,
    target_name: str = "target",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_informative, n_features),
        n_redundant=max(0, min(2, n_features - min(n_informative, n_features))),
        n_repeated=0,
        n_classes=2,
        weights=list(weights) if weights is not None else None,
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=seed,
    )
    cols = [f"x{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df[target_name] = y
    _inject_missing(df, missing_pct, cols, rng)
    _inject_outliers(df, outlier_pct, cols, rng)
    return df


def generate_dataset(
    task_type: str,
    dataset_type: str,
    *,
    n_samples: int = 500,
    noise: float = 0.1,
    n_clusters: int = 3,
    random_seed: int = 42,
    n_features: int = 20,
    n_informative: int = 5,
    target_name: str = "target",
) -> tuple[pd.DataFrame, dict]:
    task_type = task_type.lower()
    metadata = {
        "source": "generated",
        "task_type": task_type,
        "dataset_type": dataset_type,
        "random_seed": random_seed,
        "n_samples": n_samples,
        "n_features": n_features,
        "target_column": target_name if task_type in {"regression", "classification"} else "label",
    }

    if task_type == "clustering":
        mapping = {
            "Gaussian Blobs": lambda: gen_gaussian_blobs(
                n_samples=n_samples, n_clusters=n_clusters, cluster_std=max(noise, 0.01), seed=random_seed
            ),
            "Two Moons": lambda: gen_moons(n_samples=n_samples, noise=noise, seed=random_seed),
            "Concentric Circles": lambda: gen_circles(n_samples=n_samples, noise=noise, seed=random_seed),
            "Spiral": lambda: gen_spiral(
                n_samples=n_samples, n_arms=n_clusters, noise=noise, seed=random_seed
            ),
            "Anisotropic Blobs": lambda: gen_anisotropic_blobs(n_samples=n_samples, seed=random_seed),
            "Variable Density Blobs": lambda: gen_variable_density(n_samples=n_samples, seed=random_seed),
        }
        metadata["n_clusters"] = n_clusters
        metadata["noise"] = noise
        df = mapping[dataset_type]()
        df = _expand_features(df, max(n_features, 2), seed=random_seed)
        return df, metadata

    if task_type == "classification":
        mapping = {
            "Linearly Separable": lambda: gen_classification(
                n_samples=n_samples,
                n_features=max(n_features, 2),
                n_informative=min(max(n_informative, 2), max(n_features, 2)),
                seed=random_seed,
                class_sep=max(1.2 + noise, 0.5),
                flip_y=min(noise * 0.05, 0.2),
                target_name=target_name,
            ),
            "Overlapping Classes": lambda: gen_classification(
                n_samples=n_samples,
                n_features=max(n_features, 2),
                n_informative=min(max(n_informative, 2), max(n_features, 2)),
                seed=random_seed,
                class_sep=max(0.6, 1.1 - noise * 0.25),
                flip_y=min(noise * 0.12, 0.35),
                target_name=target_name,
            ),
            "Imbalanced Classes": lambda: gen_classification(
                n_samples=n_samples,
                n_features=max(n_features, 2),
                n_informative=min(max(n_informative, 2), max(n_features, 2)),
                seed=random_seed,
                class_sep=max(1.0, 1.3 - noise * 0.1),
                weights=(0.82, 0.18),
                flip_y=min(noise * 0.08, 0.25),
                target_name=target_name,
            ),
        }
        df = mapping[dataset_type]()
        metadata.update({
            "noise": noise,
            "n_informative": min(n_informative, n_features),
        })
        return df, metadata

    mapping = {
        "Linear": lambda: gen_linear_regression(n_samples=n_samples, noise=noise * 100, seed=random_seed),
        "Nonlinear": lambda: gen_nonlinear_regression(n_samples=n_samples, noise=max(noise * 20, 0.1), seed=random_seed),
        "Sinusoidal": lambda: gen_sinusoidal_regression(n_samples=n_samples, noise=noise, seed=random_seed),
        "High-Dimensional": lambda: gen_highdim_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(n_informative, n_features),
            noise=noise * 100,
            seed=random_seed,
        )[0],
    }
    df = mapping[dataset_type]()
    if task_type == "regression":
        if target_name != "target" and "target" in df.columns:
            df = df.rename(columns={"target": target_name})
    if dataset_type != "High-Dimensional":
        df = _expand_features(df, max(n_features, 1), seed=random_seed)
    metadata.update({
        "noise": noise,
        "n_features": n_features,
        "n_informative": min(n_informative, n_features),
    })
    return df, metadata
