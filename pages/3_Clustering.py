from __future__ import annotations

import matplotlib
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

from utils import api_client
from utils.api_client import ApiError
from utils.app_state import active_test_df, active_train_df, init_state
from utils.data_input import render_data_sidebar
from utils.ui import PLOTLY_THEME, PALETTE, setup_page

matplotlib.use("Agg")
import matplotlib.pyplot as plt


setup_page("DatoScope · Clustering")
init_state()
render_data_sidebar()

st.title("Clustering")
df_c = active_train_df()
if df_c is None:
    st.info("Load or generate a dataset first.")
    st.stop()
if not st.session_state.processed_run_id:
    st.info("Run Clean & Preprocess in the sidebar first — clustering reads from the warehouse via the API.")
    st.stop()
active_task = st.session_state.data_meta.get("active_ml_task", "Auto")
if active_task in {"Regression", "Classification"}:
    st.warning("The current selected ML task is not Clustering. Switch the selected ML task in the sidebar to Clustering to use this page.")
    st.stop()
if active_test_df() is not None:
    st.caption("Clustering runs on the train dataset only. Uploaded test data is kept for supervised workflows.")

feature_candidates = [c for c in df_c.select_dtypes(include="number").columns.tolist() if c not in ("label", "target")]
if len(feature_candidates) < 2:
    st.warning("Need at least 2 numeric feature columns for clustering.")
    st.stop()

st.markdown("#### Clustering Configuration")
clust_feats = st.multiselect("Feature columns", feature_candidates, default=feature_candidates[: min(5, len(feature_candidates))], key="clust_feats")

top1, top2, top3 = st.columns(3)
with top1:
    run_km = st.checkbox("K-Means", value=True)
    k_val = st.slider("K (clusters)", 2, 10, 3) if run_km else 3
with top2:
    run_db = st.checkbox("DBSCAN", value=True)
    db_eps = st.number_input("ε (eps)", min_value=0.01, max_value=10.0, value=0.5, step=0.05) if run_db else 0.5
    db_min = st.slider("min_samples", 2, 30, 5) if run_db else 5
with top3:
    run_hc = st.checkbox("Hierarchical", value=True)
    hc_k = st.slider("n_clusters", 2, 10, 3, key="hc_k") if run_hc else 3
    hc_link = st.selectbox("Linkage", ["ward", "complete", "average", "single"]) if run_hc else "ward"

clust_btn = st.button("🚀 Run Clustering", use_container_width=True)

if not clust_feats:
    st.warning("Select at least 2 feature columns.")
    st.stop()
X_c = df_c[clust_feats].dropna()
if len(X_c) < 3:
    st.warning("Not enough valid rows for clustering after removing missing values.")
    st.stop()

ground_truth_col = "label" if "label" in df_c.columns else None

if clust_btn:
    with st.spinner("Clustering…"):
        try:
            st.session_state.cluster_results = api_client.run_clustering(
                st.session_state.dataset_name,
                run_id=st.session_state.processed_run_id,
                features=clust_feats,
                run_km=run_km,
                k_val=k_val,
                run_db=run_db,
                db_eps=db_eps,
                db_min=db_min,
                run_hc=run_hc,
                hc_k=hc_k,
                hc_link=hc_link,
                ground_truth_col=ground_truth_col,
            )
        except ApiError as exc:
            st.error(f"Clustering failed: {exc}")

cres = st.session_state.cluster_results
if not cres:
    st.info("Configure and click Run Clustering to start.")
    st.stop()

results_c = cres["models"]
X_plot = pd.DataFrame.from_records(cres["X"], columns=cres["features"])
feats_c = cres["features"]

metric_rows_c = []
for name, r in results_c.items():
    metric_rows_c.append(
        {
            "Algorithm": name,
            "Clusters found": r["n_clusters"],
            "Silhouette ↑": r.get("Silhouette"),
            "Davies-Bouldin ↓": r.get("Davies_Bouldin"),
            "Calinski-Harabasz ↑": r.get("Calinski_Harabasz"),
            "FM Score ↑": r.get("FM_Score"),
            "Rand Index ↑": r.get("Rand_Index"),
        }
    )

cmdf = pd.DataFrame(metric_rows_c).set_index("Algorithm")
st.dataframe(
    cmdf.style.highlight_max(subset=["Silhouette ↑", "Calinski-Harabasz ↑", "FM Score ↑", "Rand Index ↑"], color="#10b98133")
    .highlight_min(subset=["Davies-Bouldin ↓"], color="#10b98133")
    .format(na_rep="N/A", precision=4),
    use_container_width=True,
)

coords_x, coords_y = cres["projection"]["x"], cres["projection"]["y"]
xlab, ylab = cres["projection"]["x_label"], cres["projection"]["y_label"]

for name, r in results_c.items():
    st.markdown(f"---\n#### {name} — {r['n_clusters']} cluster(s)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Clusters", r["n_clusters"])
    m2.metric("Silhouette", r.get("Silhouette") or "N/A")
    m3.metric("Davies-Bouldin", r.get("Davies_Bouldin") or "N/A")
    m4.metric("Calinski-Harabasz", r.get("Calinski_Harabasz") or "N/A")
    m5.metric("FM Score", r.get("FM_Score") or "N/A")
    m6.metric("Rand Index", r.get("Rand_Index") or "N/A")

    labels_arr = r["labels"]

    display_labels = pd.Series(labels_arr).map(
        lambda v: "Noise" if int(v) == -1 else f"Cluster {int(v) + 1}"
    )

    unique_labels = sorted(
        display_labels.unique(),
        key=lambda x: (x == "Noise", x)
    )

    color_map = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(unique_labels)}

    # ── Scatter plot — pass category_orders so Plotly respects our order ────
    plot_df = pd.DataFrame({"x": coords_x, "y": coords_y, "Cluster": display_labels})
    fig_cl = px.scatter(
        plot_df,
        x="x", y="y",
        color="Cluster",
        color_discrete_map=color_map,
        category_orders={"Cluster": unique_labels},
        labels={"x": xlab, "y": ylab},
        opacity=0.75,
    )
    fig_cl.update_layout(
        height=380,
        legend_title_text="Cluster",
        title=f"{name} — Cluster Assignments",
        **PLOTLY_THEME,
    )
    fig_cl.update_traces(marker_size=6)
    st.plotly_chart(fig_cl, use_container_width=True)

    sizes = display_labels.value_counts()
    sizes = sizes.reindex(unique_labels).fillna(0).astype(int)

    fig_sz = px.bar(
        x=sizes.index,
        y=sizes.values,
        labels={"x": "Cluster", "y": "Count"},
        color=sizes.index,
        color_discrete_map=color_map,
        category_orders={"color": unique_labels, "x": unique_labels},
    )
    fig_sz.update_layout(title="Cluster sizes", height=260, showlegend=False, **PLOTLY_THEME)
    st.plotly_chart(fig_sz, use_container_width=True)

    # Dendrogram/elbow diagnostics need the scaled feature matrix directly
    # (not a single summary metric), so they're computed here from X_plot
    # (fetched from the API alongside the clustering results) rather than
    # via a narrow one-off endpoint for each view.
    if name == "Hierarchical":
        st.markdown("##### Dendrogram (top 30 leaves)")
        fig_dend, ax = plt.subplots(figsize=(14, 4))
        fig_dend.patch.set_facecolor("#111827")
        ax.set_facecolor("#1a2235")
        Z = linkage(X_plot, method=hc_link)
        dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, color_threshold=0.6 * max(Z[:, 2]), above_threshold_color="#64748b")
        ax.tick_params(colors="#e2e8f0")
        for spine in ax.spines.values():
            spine.set_color("#1e3a5f")
        plt.tight_layout()
        st.pyplot(fig_dend, use_container_width=True)
        plt.close()

    if name == "K-Means":
        st.markdown("##### Elbow Curve (inertia)")
        inertias = []
        k_range = range(2, min(11, len(X_plot)))
        for ki in k_range:
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km.fit(X_plot)
            inertias.append(km.inertia_)
        fig_elbow = px.line(x=list(k_range), y=inertias, labels={"x": "k", "y": "Inertia"}, markers=True)
        fig_elbow.update_traces(line_color="#00d4ff", marker_color="#7c3aed", marker_size=8)
        fig_elbow.update_layout(title="Elbow Method", height=280, **PLOTLY_THEME)
        st.plotly_chart(fig_elbow, use_container_width=True)
