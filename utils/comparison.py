"""
Model comparison / winner-selection logic, shared by the API's /comparison
endpoints. Operates on plain metric dicts (as returned by the modeling/
clustering endpoints), not DataFrames-with-model-objects — framework-agnostic
so it doesn't pull in Streamlit or Plotly.
"""

from __future__ import annotations


def _normalize(values: dict[str, float], *, higher_is_better: bool) -> dict[str, float]:
    nums = [v for v in values.values() if v is not None]
    if not nums:
        return {k: 0.0 for k in values}
    lo, hi = min(nums), max(nums)
    denom = max(hi - lo, 1e-9)
    out = {}
    for k, v in values.items():
        if v is None:
            out[k] = 0.0
            continue
        norm = (v - lo) / denom
        out[k] = norm if higher_is_better else 1 - norm
    return out


def score_regression_models(results: dict[str, dict]) -> dict:
    """
    results: {model_name: {"R2": .., "CV_R2": .., "RMSE": .., "MAE": .., "Overfit_Gap": ..}}
    Mirrors pages/4_Comparison.py's Generalization Score weighting.
    """
    names = list(results.keys())
    r2_norm = _normalize({n: results[n]["R2"] for n in names}, higher_is_better=True)
    cv_r2_norm = _normalize({n: results[n]["CV_R2"] for n in names}, higher_is_better=True)
    rmse_norm = _normalize({n: results[n]["RMSE"] for n in names}, higher_is_better=False)
    mae_norm = _normalize({n: results[n]["MAE"] for n in names}, higher_is_better=False)
    gap_norm = _normalize({n: results[n]["Overfit_Gap"] for n in names}, higher_is_better=False)

    scored = {}
    for n in names:
        score = round(
            0.32 * r2_norm[n] + 0.28 * cv_r2_norm[n] + 0.16 * rmse_norm[n] + 0.12 * mae_norm[n] + 0.12 * gap_norm[n],
            4,
        )
        scored[n] = {**results[n], "generalization_score": score}

    winner = max(scored, key=lambda n: scored[n]["generalization_score"]) if scored else None
    return {"models": scored, "winner": winner, "winner_reason": "highest generalization_score (R2, CV R2, RMSE, MAE, overfit gap, weighted)"}


def score_classification_models(results: dict[str, dict]) -> dict:
    """
    results: {model_name: {"Accuracy": .., "Precision": .., "Recall": .., "F1": .., "Macro_F1": .., "CV_Accuracy": ..}}
    Winner = highest Macro F1 (matches pages/4_Comparison.py).
    """
    winner = max(results, key=lambda n: results[n]["Macro_F1"]) if results else None
    return {"models": results, "winner": winner, "winner_reason": "highest Macro_F1 (fair across class sizes)"}


def score_clustering_models(results: dict[str, dict]) -> dict:
    """
    results: {algo_name: {"Silhouette": .., "Davies_Bouldin": .., "Calinski_Harabasz": .., "FM_Score": .., "Rand_Index": ..}}
    Winner = most "metric wins" across the 5 tracked scores (matches pages/4_Comparison.py).
    """
    higher_better = ["Silhouette", "Calinski_Harabasz", "FM_Score", "Rand_Index"]
    lower_better = ["Davies_Bouldin"]
    names = list(results.keys())
    wins = {n: 0 for n in names}

    for metric in higher_better:
        valid = {n: results[n].get(metric) for n in names if results[n].get(metric) is not None}
        if valid:
            best = max(valid.values())
            for n, v in valid.items():
                if v == best:
                    wins[n] += 1

    for metric in lower_better:
        valid = {n: results[n].get(metric) for n in names if results[n].get(metric) is not None}
        if valid:
            best = min(valid.values())
            for n, v in valid.items():
                if v == best:
                    wins[n] += 1

    scored = {n: {**results[n], "metric_wins": wins[n]} for n in names}
    winner = max(scored, key=lambda n: (scored[n]["metric_wins"], scored[n].get("Silhouette") or -1)) if scored else None
    return {"models": scored, "winner": winner, "winner_reason": "most metric wins across Silhouette/Davies-Bouldin/Calinski-Harabasz/FM/Rand"}
