from __future__ import annotations

import pandas as pd
import pytest

from utils.generators import (
    CLASSIFICATION_DATASETS,
    CLUSTERING_DATASETS,
    REGRESSION_DATASETS,
    gen_anisotropic_blobs,
    gen_circles,
    gen_classification,
    gen_gaussian_blobs,
    gen_highdim_regression,
    gen_linear_regression,
    gen_moons,
    gen_nonlinear_regression,
    gen_sinusoidal_regression,
    gen_spiral,
    gen_variable_density,
    generate_dataset,
)

CLUSTERING_GENERATORS = {
    "Gaussian Blobs": lambda: gen_gaussian_blobs(n_samples=200, seed=0),
    "Two Moons": lambda: gen_moons(n_samples=200, seed=0),
    "Concentric Circles": lambda: gen_circles(n_samples=200, seed=0),
    "Spiral": lambda: gen_spiral(n_samples=200, seed=0),
    "Anisotropic Blobs": lambda: gen_anisotropic_blobs(n_samples=200, seed=0),
    "Variable Density Blobs": lambda: gen_variable_density(n_samples=200, seed=0),
}


class TestClusteringGenerators:
    @pytest.mark.parametrize("name", CLUSTERING_DATASETS)
    def test_shape_and_label_column(self, name):
        df = CLUSTERING_GENERATORS[name]()
        assert isinstance(df, pd.DataFrame)
        assert "label" in df.columns
        assert len(df) > 0
        assert {"x1", "x2"}.issubset(df.columns)

    def test_missing_and_outliers_injected(self):
        df = gen_gaussian_blobs(n_samples=500, seed=0, missing_pct=0.1, outlier_pct=0.05)
        assert df[["x1", "x2"]].isnull().sum().sum() > 0

    def test_reproducible_with_same_seed(self):
        df1 = gen_gaussian_blobs(n_samples=200, seed=42)
        df2 = gen_gaussian_blobs(n_samples=200, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestRegressionGenerators:
    def test_linear_regression_shape(self):
        df = gen_linear_regression(n_samples=300, seed=0)
        assert {"x1", "target"}.issubset(df.columns)
        assert len(df) == 300

    def test_nonlinear_regression_shape(self):
        df = gen_nonlinear_regression(n_samples=300, seed=0)
        assert {"x1", "target"}.issubset(df.columns)

    def test_sinusoidal_regression_shape(self):
        df = gen_sinusoidal_regression(n_samples=300, seed=0)
        assert {"x1", "target"}.issubset(df.columns)

    def test_highdim_regression_returns_coef_df(self):
        df, coef_df = gen_highdim_regression(n_samples=200, n_features=10, n_informative=3, seed=0)
        assert df.shape[1] == 11  # 10 features + target
        assert len(coef_df) == 10
        assert {"feature", "true_coefficient"}.issubset(coef_df.columns)


class TestClassificationGenerator:
    def test_shape_and_binary_target(self):
        df = gen_classification(n_samples=300, n_features=5, n_informative=3, seed=0)
        assert df.shape[1] == 6  # 5 features + target
        assert set(df["target"].dropna().unique()) <= {0, 1}

    def test_custom_target_name(self):
        df = gen_classification(n_samples=100, n_features=3, n_informative=2, seed=0, target_name="label")
        assert "label" in df.columns and "target" not in df.columns

    def test_imbalanced_weights_produce_skewed_classes(self):
        df = gen_classification(
            n_samples=1000, n_features=5, n_informative=3, seed=0, weights=(0.9, 0.1), flip_y=0.0
        )
        counts = df["target"].value_counts(normalize=True)
        assert counts.max() > 0.75  # majority class should dominate


class TestGenerateDatasetDispatcher:
    @pytest.mark.parametrize("dataset_type", CLUSTERING_DATASETS)
    def test_clustering_dispatch(self, dataset_type):
        df, meta = generate_dataset("Clustering", dataset_type, n_samples=150, n_features=4, random_seed=0)
        assert meta["task_type"] == "clustering"
        assert meta["target_column"] == "label"
        assert "label" in df.columns
        # requested 4 features -> expanded beyond the base x1/x2
        assert len([c for c in df.columns if c.startswith("x")]) >= 4

    @pytest.mark.parametrize("dataset_type", REGRESSION_DATASETS)
    def test_regression_dispatch(self, dataset_type):
        df, meta = generate_dataset(
            "Regression", dataset_type, n_samples=150, n_features=6, n_informative=3, random_seed=0
        )
        assert meta["task_type"] == "regression"
        assert "target" in df.columns

    @pytest.mark.parametrize("dataset_type", CLASSIFICATION_DATASETS)
    def test_classification_dispatch(self, dataset_type):
        df, meta = generate_dataset(
            "Classification", dataset_type, n_samples=150, n_features=5, n_informative=3, random_seed=0
        )
        assert meta["task_type"] == "classification"
        assert "target" in df.columns
        assert set(df["target"].dropna().unique()) <= {0, 1}

    def test_custom_target_name_propagates_for_regression(self):
        df, meta = generate_dataset("Regression", "Linear", n_samples=100, target_name="y", random_seed=0)
        assert "y" in df.columns
        assert meta["target_column"] == "y"

    def test_metadata_has_source_generated(self):
        _, meta = generate_dataset("Regression", "Linear", n_samples=50, random_seed=0)
        assert meta["source"] == "generated"

    def test_unknown_dataset_type_raises_keyerror(self):
        with pytest.raises(KeyError):
            generate_dataset("Regression", "Not A Real Type", n_samples=50)
