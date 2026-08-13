from __future__ import annotations

import pickle

import pandas as pd
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression

from utils.modeling import (
    export_model_bytes,
    infer_supervised_task,
    projection_for_plot,
    run_classification_models,
    run_clustering_models,
    run_regression_models,
)


@pytest.fixture
def regression_df():
    X, y = make_regression(n_samples=200, n_features=3, noise=5.0, random_state=0)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["target"] = y
    return df


@pytest.fixture
def classification_df():
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3, n_redundant=0, random_state=0)
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    df["target"] = y
    return df


@pytest.fixture
def clustering_df():
    X, y = make_blobs(n_samples=150, centers=3, n_features=2, random_state=0)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    return df


class TestInferSupervisedTask:
    def test_numeric_continuous_is_regression(self, regression_df):
        assert infer_supervised_task(regression_df, "target") == "Regression"

    def test_binary_numeric_is_classification(self, classification_df):
        assert infer_supervised_task(classification_df, "target") == "Classification"

    def test_string_target_is_classification(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "target": ["x", "y", "x", "y"]})
        assert infer_supervised_task(df, "target") == "Classification"


class TestExportModelBytes:
    def test_roundtrip(self):
        from sklearn.linear_model import LinearRegression

        model = LinearRegression().fit([[1], [2], [3]], [1, 2, 3])
        payload = export_model_bytes(model, features=["x1"], target="y", task_type="Regression")
        restored = pickle.loads(payload)
        assert restored["features"] == ["x1"]
        assert restored["target"] == "y"
        assert restored["task_type"] == "Regression"
        assert restored["model"].predict([[4]])[0] == pytest.approx(4.0, abs=0.5)


class TestRunRegressionModels:
    def test_returns_selected_models_only(self, regression_df):
        results = run_regression_models(
            regression_df,
            None,
            features=["x1", "x2", "x3"],
            target_col="target",
            test_size=0.2,
            cv_folds=3,
            random_seed=0,
            run_lr=True,
            run_ridge=False,
            ridge_alpha=1.0,
            run_lasso=False,
            lasso_alpha=0.1,
        )
        assert list(results.keys()) == ["Linear Regression"]

    def test_metrics_present_and_reasonable(self, regression_df):
        results = run_regression_models(
            regression_df,
            None,
            features=["x1", "x2", "x3"],
            target_col="target",
            test_size=0.2,
            cv_folds=3,
            random_seed=0,
            run_lr=True,
            run_ridge=True,
            ridge_alpha=1.0,
            run_lasso=True,
            lasso_alpha=0.1,
        )
        assert set(results.keys()) == {"Linear Regression", "Ridge (α=1.0)", "Lasso (α=0.1)"}
        for row in results.values():
            for key in ("R²", "MSE", "RMSE", "MAE", "CV R²", "Overfit Gap"):
                assert key in row
            assert row["R²"] > 0.5  # make_regression data is easy to fit well
            assert "coef" in row  # all three models are linear

    def test_explicit_test_df_used(self, regression_df):
        train, test = regression_df.iloc[:150], regression_df.iloc[150:]
        results = run_regression_models(
            train,
            test,
            features=["x1", "x2", "x3"],
            target_col="target",
            test_size=0.2,
            cv_folds=3,
            random_seed=0,
            run_lr=True,
            run_ridge=False,
            ridge_alpha=1.0,
            run_lasso=False,
            lasso_alpha=0.1,
        )
        row = results["Linear Regression"]
        assert row["split_method"] == "Uploaded test file"
        assert len(row["y_test"]) == len(test)


class TestRunClassificationModels:
    def test_metrics_and_confusion_matrix(self, classification_df):
        results = run_classification_models(
            classification_df,
            None,
            features=["x1", "x2", "x3", "x4"],
            target_col="target",
            test_size=0.2,
            cv_folds=3,
            random_seed=0,
            run_logreg=True,
            run_rf=True,
            rf_estimators=50,
            rf_max_depth=5,
            rf_min_samples_split=2,
            rf_min_samples_leaf=1,
            rf_max_features="sqrt",
            rf_bootstrap=True,
            run_knn=False,
            knn_neighbors=5,
        )
        assert set(results.keys()) == {"Logistic Regression", "Random Forest (50)"}
        for row in results.values():
            for key in ("Accuracy", "Precision", "Recall", "F1", "Macro F1", "CV Accuracy"):
                assert 0.0 <= row[key] <= 1.0
            cm = row["Confusion Matrix"]
            assert cm.shape == (2, 2)
            assert cm.sum() == len(row["y_test"])

        rf_model = results["Random Forest (50)"]["model"]
        assert hasattr(rf_model, "estimators_")
        assert len(rf_model.estimators_) == 50


class TestRunClusteringModels:
    def test_structure_without_ground_truth(self, clustering_df):
        result = run_clustering_models(
            clustering_df,
            features=["x1", "x2"],
            run_km=True,
            k_val=3,
            run_db=False,
            db_eps=0.5,
            db_min=5,
            run_hc=False,
            hc_k=3,
            hc_link="ward",
        )
        assert "K-Means" in result["results"]
        km = result["results"]["K-Means"]
        assert km["n_clusters"] == 3
        assert km["Silhouette"] > 0.3  # make_blobs clusters are well separated
        assert "FM Score" not in km

    def test_ground_truth_adds_external_metrics(self, clustering_df):
        result = run_clustering_models(
            clustering_df,
            features=["x1", "x2"],
            run_km=True,
            k_val=3,
            run_db=False,
            db_eps=0.5,
            db_min=5,
            run_hc=False,
            hc_k=3,
            hc_link="ward",
            ground_truth=clustering_df["label"],
        )
        km = result["results"]["K-Means"]
        assert "FM Score" in km and "Rand Index" in km
        assert km["Rand Index"] > 0.8  # blobs are trivially separable, expect near-perfect recovery


class TestProjectionForPlot:
    def test_two_features_no_pca(self, clustering_df):
        coords, xlab, ylab = projection_for_plot(clustering_df, ["x1", "x2"])
        assert coords.shape == (len(clustering_df), 2)
        assert xlab == "x1" and ylab == "x2"

    def test_more_than_two_features_uses_pca(self, regression_df):
        coords, xlab, ylab = projection_for_plot(regression_df, ["x1", "x2", "x3"])
        assert coords.shape == (len(regression_df), 2)
        assert xlab.startswith("PC1")
        assert ylab.startswith("PC2")
