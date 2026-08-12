from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.app_state import init_state
from utils.data_input import render_data_sidebar
from utils.ui import PLOTLY_THEME, hex_to_rgba, setup_page


def score_regression_models(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    for col in ["R²", "CV R²"]:
        max_v = scored[col].max()
        min_v = scored[col].min()
        denom = max(max_v - min_v, 1e-9)
        scored[f"{col}_norm"] = (scored[col] - min_v) / denom
    for col in ["RMSE", "MAE", "Overfit Gap"]:
        max_v = scored[col].max()
        min_v = scored[col].min()
        denom = max(max_v - min_v, 1e-9)
        scored[f"{col}_norm"] = 1 - ((scored[col] - min_v) / denom)
    scored["Generalization Score"] = (
        0.32 * scored["R²_norm"]
        + 0.28 * scored["CV R²_norm"]
        + 0.16 * scored["RMSE_norm"]
        + 0.12 * scored["MAE_norm"]
        + 0.12 * scored["Overfit Gap_norm"]
    ).round(4)
    return scored


def score_clustering_models(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["Metric Wins"] = 0

    higher_better = ["Silhouette", "Calinski-Harabasz", "FM Score", "Rand Index"]
    lower_better = ["Davies-Bouldin"]

    for metric in higher_better:
        valid = scored[metric].dropna()
        if not valid.empty:
            best_value = valid.max()
            scored.loc[scored[metric] == best_value, "Metric Wins"] += 1

    for metric in lower_better:
        valid = scored[metric].dropna()
        if not valid.empty:
            best_value = valid.min()
            scored.loc[scored[metric] == best_value, "Metric Wins"] += 1

    return scored


setup_page("DatoScope · Comparison")
init_state()
render_data_sidebar()

st.title("Model Comparison")

reg_res = st.session_state.reg_results
cls_res = st.session_state.cls_results
clust_res = st.session_state.cluster_results.get("results", {})

if not reg_res and not cls_res and not clust_res:
    st.info("Train supervised or clustering models to compare them here.")
    st.stop()

if reg_res:
    st.markdown("### Regression Models")
    st.caption("Best regression model is chosen using a Generalization Score that combines test R², cross-validated R², RMSE, MAE, and overfitting gap.")
    rdf = pd.DataFrame(
        [
            {
                "Model": name,
                "R²": r["R²"],
                "CV R²": r["CV R²"],
                "RMSE": r["RMSE"],
                "MAE": r["MAE"],
                "Overfit Gap": r["Overfit Gap"],
            }
            for name, r in reg_res.items()
        ]
    )
    scored_rdf = score_regression_models(rdf)
    best_r = scored_rdf.loc[scored_rdf["Generalization Score"].idxmax()]
    st.markdown(
        f"""
        <div class="winner-banner">
          <div style="font-size:11px;color:#10b981;letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:4px;">Best Regression Model</div>
          <div style="font-size:1.4rem;font-weight:700;font-family:'Space Mono',monospace;color:#fff;">🥇 {best_r['Model']}</div>
          <div style="color:#94a3b8;font-size:13px;margin-top:6px;">Generalization Score = {best_r['Generalization Score']} · R² = {best_r['R²']} · CV R² = {best_r['CV R²']}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="ds-card">
          <b>Why this model won</b><br>
          It has the highest <code>Generalization Score</code> of <code>{best_r['Generalization Score']}</code>, which rewards strong predictive fit and stable generalization together.
          In this run it combines test <code>R² = {best_r['R²']}</code>, cross-validated <code>R² = {best_r['CV R²']}</code>, low error levels, and a controlled <code>Overfit Gap = {best_r['Overfit Gap']}</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(scored_rdf[["Model", "R²", "CV R²", "RMSE", "MAE", "Overfit Gap", "Generalization Score"]], use_container_width=True)
    fig_r = px.bar(scored_rdf, x="Model", y=["R²", "CV R²", "Generalization Score"], barmode="group", color_discrete_sequence=["#00d4ff", "#7c3aed", "#10b981"])
    fig_r.update_layout(title="Regression Comparison", height=340, legend_title_text="Metric", **PLOTLY_THEME)
    st.plotly_chart(fig_r, use_container_width=True)

if cls_res:
    st.markdown("### Classification Models")
    st.caption("Best classification model is chosen by highest Macro F1, which treats all classes more fairly than plain accuracy when class sizes differ.")
    cdf = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": r["Accuracy"],
                "Precision": r["Precision"],
                "Recall": r["Recall"],
                "F1": r["F1"],
                "Macro F1": r["Macro F1"],
                "CV Accuracy": r["CV Accuracy"],
            }
            for name, r in cls_res.items()
        ]
    )
    best_c = cdf.loc[cdf["Macro F1"].idxmax()]
    st.markdown(
        f"""
        <div class="winner-banner">
          <div style="font-size:11px;color:#10b981;letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:4px;">Best Classification Model</div>
          <div style="font-size:1.4rem;font-weight:700;font-family:'Space Mono',monospace;color:#fff;">🥇 {best_c['Model']}</div>
          <div style="color:#94a3b8;font-size:13px;margin-top:6px;">Macro F1 = {best_c['Macro F1']} · Accuracy = {best_c['Accuracy']} · CV Accuracy = {best_c['CV Accuracy']}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="ds-card">
          <b>Why this model won</b><br>
          It has the highest <code>Macro F1 = {best_c['Macro F1']}</code>, which is the primary selection metric on this page because it gives equal importance to each class.
          Its <code>Accuracy = {best_c['Accuracy']}</code>, weighted <code>F1 = {best_c['F1']}</code>, and <code>CV Accuracy = {best_c['CV Accuracy']}</code> are shown as supporting signals so you can judge overall correctness and stability too.
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig_c = px.bar(cdf, x="Model", y=["Macro F1", "Accuracy", "CV Accuracy"], barmode="group")
    fig_c.update_layout(title="Classification Metrics", height=340, **PLOTLY_THEME)
    st.plotly_chart(fig_c, use_container_width=True)

if clust_res:
    st.markdown("### Clustering Models")
    st.caption("Best clustering model is chosen by the algorithm that wins the largest number of available clustering metrics across the five tracked scores.")
    cldf = pd.DataFrame(
        [
            {
                "Algorithm": name,
                "Clusters": r["n_clusters"],
                "Silhouette": r.get("Silhouette"),
                "Davies-Bouldin": r.get("Davies-Bouldin"),
                "Calinski-Harabasz": r.get("Calinski-Harabasz"),
                "FM Score": r.get("FM Score"),
                "Rand Index": r.get("Rand Index"),
            }
            for name, r in clust_res.items()
        ]
    )
    scored_cldf = score_clustering_models(cldf)
    valid = scored_cldf.dropna(subset=["Metric Wins"])
    if not valid.empty:
        best_cluster = valid.sort_values(
            by=["Metric Wins", "Silhouette", "Rand Index", "FM Score"],
            ascending=[False, False, False, False],
            na_position="last",
        ).iloc[0]
        st.markdown(
            f"""
            <div class="winner-banner">
              <div style="font-size:11px;color:#10b981;letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:4px;">Best Clustering Algorithm</div>
              <div style="font-size:1.4rem;font-weight:700;font-family:'Space Mono',monospace;color:#fff;">🥇 {best_cluster['Algorithm']}</div>
              <div style="color:#94a3b8;font-size:13px;margin-top:6px;">Metric Wins = {best_cluster['Metric Wins']} · Silhouette = {best_cluster['Silhouette']} · FM = {best_cluster['FM Score']} · Rand = {best_cluster['Rand Index']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="ds-card">
              <b>Why this model won</b><br>
              It wins the highest number of clustering metrics with <code>{best_cluster['Metric Wins']}</code> metric wins across the tracked scores.
              The comparison checks five metrics when available:
              higher <code>Silhouette</code>, lower <code>Davies-Bouldin</code>, higher <code>Calinski-Harabasz</code>, higher <code>FM Score</code>, and higher <code>Rand Index</code>.
              Ties are broken using stronger separation-oriented scores such as <code>Silhouette</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.dataframe(scored_cldf, use_container_width=True)

if reg_res:
    st.markdown("### Regression Radar")
    st.caption("This radar chart compares regression models across normalized metrics. Larger filled areas indicate a better balance of predictive power, lower error, and better generalization.")
    try:
        cats = ["R²", "CV R²", "RMSE (inv)", "MAE (inv)", "Gap (inv)"]
        fig_rad = go.Figure()
        colors_ = ["#00d4ff", "#7c3aed", "#10b981", "#f59e0b"]
        for i, (_, row_) in enumerate(scored_rdf.iterrows()):
            vals = [
                row_["R²_norm"],
                row_["CV R²_norm"],
                row_["RMSE_norm"],
                row_["MAE_norm"],
                row_["Overfit Gap_norm"],
            ]
            vals += [vals[0]]
            color = colors_[i % len(colors_)]
            fig_rad.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=cats + [cats[0]],
                    fill="toself",
                    name=row_["Model"],
                    line_color=color,
                    fillcolor=hex_to_rgba(color, 0.22),
                )
            )
        fig_rad.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#64748b"), gridcolor="#1e3a5f"),
                angularaxis=dict(tickfont=dict(color="#e2e8f0"), gridcolor="#1e3a5f"),
                bgcolor="#111827",
            ),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_rad, use_container_width=True)
    except Exception as exc:
        st.error(f"Regression radar could not be rendered: {exc}")
