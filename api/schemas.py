"""Pydantic request models for the DatoScope API."""

from __future__ import annotations

from pydantic import BaseModel


class CleanParams(BaseModel):
    missing_strategy: str = "mean"  # mean | median | mode | drop
    outlier_method: str = "IQR"  # IQR | Z-Score | none
    scale_method: str = "Standard"  # Standard | MinMax | Robust
    remove_dupes: bool = True
    label_col: str | None = None
    encode_categoricals: bool = False
    categorical_encoding: str = "One-Hot"
    categorical_encoding_map: dict[str, str] | None = None


class GenerateDatasetRequest(BaseModel):
    dataset_name: str | None = None
    task_type: str  # Regression | Classification | Clustering
    dataset_type: str
    n_samples: int = 500
    noise: float = 0.15
    n_clusters: int = 3
    random_seed: int = 42
    n_features: int = 6
    n_informative: int = 5
    target_name: str = "target"
    create_split: bool = False  # Regression/Classification only; splits into two datasets (train + "<name>__test")
    test_split_pct: int = 20


class KaggleDatasetRequest(BaseModel):
    handle: str
    dataset_name: str | None = None


class CleanRequest(CleanParams):
    run_id: str | None = None  # raw run to transform; defaults to the latest raw extract


class RegressionRequest(BaseModel):
    run_id: str | None = None
    test_dataset_name: str | None = None  # if set, evaluate against this dataset instead of an auto split
    test_run_id: str | None = None
    features: list[str]
    target_col: str
    test_size: float = 0.2
    cv_folds: int = 5
    random_seed: int = 42
    run_lr: bool = True
    run_ridge: bool = True
    ridge_alpha: float = 1.0
    run_lasso: bool = True
    lasso_alpha: float = 0.1


class ClassificationRequest(BaseModel):
    run_id: str | None = None
    test_dataset_name: str | None = None
    test_run_id: str | None = None
    features: list[str]
    target_col: str
    test_size: float = 0.2
    cv_folds: int = 5
    random_seed: int = 42
    run_logreg: bool = True
    run_rf: bool = True
    rf_estimators: int = 200
    rf_max_depth: int | None = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_max_features: str = "sqrt"
    rf_bootstrap: bool = True
    run_knn: bool = True
    knn_neighbors: int = 5


class ClusteringRequest(BaseModel):
    run_id: str | None = None
    features: list[str]
    run_km: bool = True
    k_val: int = 3
    run_db: bool = True
    db_eps: float = 0.5
    db_min: int = 5
    run_hc: bool = True
    hc_k: int = 3
    hc_link: str = "ward"
    ground_truth_col: str | None = None


class ComparisonRequest(BaseModel):
    results: dict
