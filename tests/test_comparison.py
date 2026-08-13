from __future__ import annotations

from utils.comparison import score_classification_models, score_clustering_models, score_regression_models


class TestScoreRegressionModels:
    def test_picks_best_on_all_metrics(self):
        results = {
            "good": {"R2": 0.95, "CV_R2": 0.94, "RMSE": 1.0, "MAE": 0.8, "Overfit_Gap": 0.01},
            "bad": {"R2": 0.5, "CV_R2": 0.45, "RMSE": 10.0, "MAE": 8.0, "Overfit_Gap": 0.3},
        }
        out = score_regression_models(results)
        assert out["winner"] == "good"
        assert out["models"]["good"]["generalization_score"] > out["models"]["bad"]["generalization_score"]

    def test_empty_results(self):
        out = score_regression_models({})
        assert out["winner"] is None
        assert out["models"] == {}

    def test_single_model_wins_trivially(self):
        results = {"only": {"R2": 0.7, "CV_R2": 0.7, "RMSE": 2.0, "MAE": 1.5, "Overfit_Gap": 0.05}}
        out = score_regression_models(results)
        assert out["winner"] == "only"


class TestScoreClassificationModels:
    def test_picks_highest_macro_f1(self):
        results = {
            "a": {"Accuracy": 0.9, "Macro_F1": 0.7},
            "b": {"Accuracy": 0.8, "Macro_F1": 0.85},
        }
        out = score_classification_models(results)
        assert out["winner"] == "b"

    def test_empty_results(self):
        out = score_classification_models({})
        assert out["winner"] is None


class TestScoreClusteringModels:
    def test_metric_wins_counted_correctly(self):
        results = {
            "A": {"Silhouette": 0.9, "Davies_Bouldin": 0.5, "Calinski_Harabasz": 100, "FM_Score": 0.8, "Rand_Index": 0.8},
            "B": {"Silhouette": 0.5, "Davies_Bouldin": 0.2, "Calinski_Harabasz": 50, "FM_Score": 0.5, "Rand_Index": 0.5},
        }
        out = score_clustering_models(results)
        # A wins Silhouette, Calinski-Harabasz, FM, Rand (4); B wins only Davies-Bouldin (1)
        assert out["models"]["A"]["metric_wins"] == 4
        assert out["models"]["B"]["metric_wins"] == 1
        assert out["winner"] == "A"

    def test_missing_metrics_handled_gracefully(self):
        results = {
            "A": {"Silhouette": 0.9, "Davies_Bouldin": None, "Calinski_Harabasz": None, "FM_Score": None, "Rand_Index": None},
            "B": {"Silhouette": 0.5, "Davies_Bouldin": None, "Calinski_Harabasz": None, "FM_Score": None, "Rand_Index": None},
        }
        out = score_clustering_models(results)
        assert out["winner"] == "A"
        assert out["models"]["A"]["metric_wins"] == 1

    def test_tie_in_metric_wins_broken_by_silhouette(self):
        # A wins Silhouette only, B wins Davies-Bouldin only -> tied at 1 metric win each.
        results = {
            "A": {"Silhouette": 0.9, "Davies_Bouldin": 0.9, "Calinski_Harabasz": None, "FM_Score": None, "Rand_Index": None},
            "B": {"Silhouette": 0.3, "Davies_Bouldin": 0.1, "Calinski_Harabasz": None, "FM_Score": None, "Rand_Index": None},
        }
        out = score_clustering_models(results)
        assert out["models"]["A"]["metric_wins"] == out["models"]["B"]["metric_wins"] == 1
        assert out["winner"] == "A"  # tiebreak: higher Silhouette
