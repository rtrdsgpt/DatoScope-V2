from __future__ import annotations

import pickle

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import plot_tree

from utils import api_client
from utils.api_client import ApiError
from utils.app_state import active_test_df, active_train_df, init_state
from utils.data_input import render_data_sidebar
from utils.modeling import infer_supervised_task
from utils.ui import PLOTLY_THEME, setup_page

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@st.cache_data(show_spinner=False)
def _download_and_unpickle(model_id: str) -> dict:
    return pickle.loads(api_client.download_model(model_id))


setup_page("DatoScope · Supervised")
init_state()
render_data_sidebar()

st.title("Supervised Modeling")

df_train = active_train_df()
df_test = active_test_df()
if df_train is None:
    st.info("Load or generate a dataset first.")
    st.stop()
if not st.session_state.processed_run_id:
    st.info("Run Clean & Preprocess in the sidebar first — modeling reads from the warehouse via the API.")
    st.stop()

active_task = st.session_state.data_meta.get("active_ml_task", "Auto")
if active_task == "Clustering":
    st.warning("The current selected ML task is Clustering. Switch the selected ML task in the sidebar to Regression or Classification to use this page.")
    st.stop()

if len(df_train.columns) < 2:
    st.warning("Need at least two columns for supervised modeling.")
    st.stop()

allowed_supervised = ["Regression", "Classification"] if active_task == "Auto" else [active_task]
task_choice = st.radio("Choose supervised task", allowed_supervised, horizontal=True)
all_targets = df_train.columns.tolist()

if task_choice == "Regression":
    default_reg = st.session_state.regression_target if st.session_state.regression_target in all_targets else all_targets[-1]
    target_col = st.selectbox("🎯 Regression target", all_targets, index=all_targets.index(default_reg), key="regression_target_select")
    st.session_state.regression_target = target_col
else:
    default_cls = st.session_state.classification_target if st.session_state.classification_target in all_targets else all_targets[-1]
    target_col = st.selectbox("🎯 Classification target", all_targets, index=all_targets.index(default_cls), key="classification_target_select")
    st.session_state.classification_target = target_col

st.session_state.data_meta["target_column"] = target_col
st.caption(f"Suggested task from target type: {infer_supervised_task(df_train, target_col)}")

feature_candidates = [c for c in df_train.columns if c != target_col]
default_features = [c for c in feature_candidates if pd.api.types.is_numeric_dtype(df_train[c])]
features = st.multiselect("Feature columns (X)", feature_candidates, default=default_features[: min(len(default_features), 10)])

settings1, settings2, settings3 = st.columns(3)
with settings1:
    test_size = st.slider("Test split %", 10, 40, 20) / 100
with settings2:
    cv_folds = st.slider("CV folds", 3, 10, 5)
with settings3:
    random_seed = st.number_input("Split seed", min_value=0, value=42, step=1)

if not features:
    st.warning("Select at least one feature column.")
    st.stop()

has_test_dataset = bool(st.session_state.test_processed_run_id)
missing_in_test = [c for c in features if df_test is not None and c not in df_test.columns]
if missing_in_test:
    st.error(f"Uploaded test data is missing features: {', '.join(missing_in_test)}")
    st.stop()
if df_test is not None and target_col not in df_test.columns:
    st.error("Target column is missing from the uploaded test dataset.")
    st.stop()

numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df_train[c])]
if len(numeric_features) != len(features):
    st.warning("Non-numeric features selected. Current supervised models use numeric features only.")
    features = numeric_features
    if not features:
        st.stop()

test_kwargs = (
    {"test_dataset_name": st.session_state.test_dataset_name, "test_run_id": st.session_state.test_processed_run_id}
    if has_test_dataset
    else {}
)

if task_choice == "Regression":
    st.markdown("#### Regression Models")
    conf_a, conf_b = st.columns(2)
    with conf_a:
        run_lr = st.checkbox("Linear Regression", value=True)
        run_ridge = st.checkbox("Ridge", value=True)
        ridge_alpha = st.select_slider("Ridge α", [0.001, 0.01, 0.1, 1, 10, 100], value=1.0)
    with conf_b:
        run_lasso = st.checkbox("Lasso", value=True)
        lasso_alpha = st.select_slider("Lasso α", [0.001, 0.01, 0.1, 1, 10, 100], value=0.1)

    if st.button("🚀 Train Regression Models", use_container_width=True):
        with st.spinner("Training…"):
            try:
                st.session_state.reg_results = api_client.train_regression(
                    st.session_state.dataset_name,
                    run_id=st.session_state.processed_run_id,
                    features=features,
                    target_col=target_col,
                    test_size=test_size,
                    cv_folds=cv_folds,
                    random_seed=random_seed,
                    run_lr=run_lr,
                    run_ridge=run_ridge,
                    ridge_alpha=ridge_alpha,
                    run_lasso=run_lasso,
                    lasso_alpha=lasso_alpha,
                    **test_kwargs,
                )
            except ApiError as exc:
                st.error(f"Training failed: {exc}")
            else:
                if st.session_state.reg_results:
                    st.session_state.data_meta["split_method"] = next(iter(st.session_state.reg_results.values()))["split_method"]

    res = st.session_state.reg_results
    if not res:
        st.info("Configure and run regression models.")
        st.stop()

    st.caption(f"Evaluation split: {next(iter(res.values()))['split_method']}")
    metric_rows = [
        {
            "Model": name,
            "R²": r["R2"],
            "CV R²": r["CV_R2"],
            "Overfit Gap": r["Overfit_Gap"],
            "RMSE": r["RMSE"],
            "MAE": r["MAE"],
            "MSE": r["MSE"],
        }
        for name, r in res.items()
    ]
    mdf = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(
        mdf.style.highlight_max(subset=["R²", "CV R²"], color="#10b98133")
        .highlight_min(subset=["Overfit Gap", "RMSE", "MAE", "MSE"], color="#10b98133")
        .format(precision=4),
        use_container_width=True,
    )

    for name, r in res.items():
        st.markdown(f"---\n#### {name}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R²", r["R2"])
        m2.metric("CV R²", r["CV_R2"])
        m3.metric("RMSE", r["RMSE"])
        m4.metric("Overfit Gap", r["Overfit_Gap"])

        ytest_arr = np.array(r["y_test"])
        ypred_arr = np.array(r["y_pred"])
        residuals = ytest_arr - ypred_arr
        min_v, max_v = float(ytest_arr.min()), float(ytest_arr.max())
        fa, fb, fc = st.columns(3)

        with fa:
            fig_ap = go.Figure()
            fig_ap.add_trace(
                go.Scatter(
                    x=ytest_arr,
                    y=ypred_arr,
                    mode="markers",
                    name="Predicted points",
                    marker=dict(color="#00d4ff", opacity=0.6, size=5),
                )
            )
            fig_ap.add_trace(
                go.Scatter(
                    x=[min_v, max_v],
                    y=[min_v, max_v],
                    mode="lines",
                    name="Ideal prediction line",
                    line=dict(color="#7c3aed", dash="dash"),
                )
            )
            fig_ap.update_layout(
                title="Actual vs Predicted",
                xaxis_title="Actual",
                yaxis_title="Predicted",
                height=300,
                **PLOTLY_THEME,
            )
            st.plotly_chart(fig_ap, use_container_width=True)

        with fb:
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(x=residuals, nbinsx=30, marker_color="#7c3aed", opacity=0.8))
            fig_res.update_layout(title="Residuals", xaxis_title="Residual", yaxis_title="Count", height=300, **PLOTLY_THEME)
            st.plotly_chart(fig_res, use_container_width=True)

        with fc:
            if r.get("coef"):
                coef_df = pd.DataFrame({"Feature": list(r["coef"].keys()), "Coefficient": list(r["coef"].values())})
                coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=True).index)
                fig_c2 = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", color="Coefficient", color_continuous_scale="RdBu", color_continuous_midpoint=0)
                fig_c2.update_layout(title="Coefficients", height=300, coloraxis_showscale=False, **PLOTLY_THEME)
                st.plotly_chart(fig_c2, use_container_width=True)

        try:
            model_bytes = api_client.download_model(r["model_id"])
        except ApiError as exc:
            st.caption(f"Model download unavailable: {exc}")
        else:
            st.download_button(
                f"⬇ Download {name} model",
                model_bytes,
                file_name=f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('α', 'alpha')}.pkl",
                mime="application/octet-stream",
                key=f"download_reg_{name}",
            )

else:
    st.markdown("#### Classification Models")
    st.markdown("#### Class Distribution")
    class_counts = (
        df_train[target_col]
        .dropna()
        .astype(str)
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Count")
    )
    class_counts["Percent"] = ((class_counts["Count"] / class_counts["Count"].sum()) * 100).round(2)
    cc1, cc2 = st.columns([0.95, 1.05])
    with cc1:
        st.dataframe(class_counts, use_container_width=True, hide_index=True)
    with cc2:
        fig_class_count = px.bar(
            class_counts,
            x="Class",
            y="Count",
            color="Class",
            text="Count",
            title="Class Count in Training Data",
        )
        fig_class_count.update_traces(textposition="outside")
        fig_class_count.update_layout(height=320, showlegend=False, **PLOTLY_THEME)
        st.plotly_chart(fig_class_count, use_container_width=True)
    top_a, top_b = st.columns(2)
    with top_a:
        run_logreg = st.checkbox("Logistic Regression", value=True)
        run_knn = st.checkbox("KNN", value=False)
        knn_neighbors = st.slider("KNN neighbors", 1, 25, 5)
    with top_b:
        run_rf = st.checkbox("Random Forest", value=True)
        rf_estimators = st.slider("RF estimators", 50, 500, 200, step=50)
        rf_max_depth_raw = st.slider("RF max depth (0 = None)", 0, 30, 8)
        rf_min_samples_split = st.slider("RF min samples split", 2, 20, 2)
        rf_min_samples_leaf = st.slider("RF min samples leaf", 1, 20, 1)
        rf_max_features = st.selectbox("RF max features", ["sqrt", "log2"])
        rf_bootstrap = st.checkbox("RF bootstrap", value=True)
    rf_max_depth = None if rf_max_depth_raw == 0 else rf_max_depth_raw

    if st.button("🚀 Train Classification Models", use_container_width=True):
        with st.spinner("Training…"):
            try:
                st.session_state.cls_results = api_client.train_classification(
                    st.session_state.dataset_name,
                    run_id=st.session_state.processed_run_id,
                    features=features,
                    target_col=target_col,
                    test_size=test_size,
                    cv_folds=cv_folds,
                    random_seed=random_seed,
                    run_logreg=run_logreg,
                    run_rf=run_rf,
                    rf_estimators=rf_estimators,
                    rf_max_depth=rf_max_depth,
                    rf_min_samples_split=rf_min_samples_split,
                    rf_min_samples_leaf=rf_min_samples_leaf,
                    rf_max_features=rf_max_features,
                    rf_bootstrap=rf_bootstrap,
                    run_knn=run_knn,
                    knn_neighbors=knn_neighbors,
                    **test_kwargs,
                )
            except ApiError as exc:
                st.error(f"Training failed: {exc}")
            else:
                if st.session_state.cls_results:
                    st.session_state.data_meta["split_method"] = next(iter(st.session_state.cls_results.values()))["split_method"]

    res = st.session_state.cls_results
    if not res:
        st.info("Configure and run classification models.")
        st.stop()

    st.caption(f"Evaluation split: {next(iter(res.values()))['split_method']}")
    metric_rows = [
        {
            "Model": name,
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1": r["F1"],
            "Macro F1": r["Macro_F1"],
            "CV Accuracy": r["CV_Accuracy"],
        }
        for name, r in res.items()
    ]
    cdf = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(
        cdf.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1", "Macro F1", "CV Accuracy"], color="#10b98133").format(precision=4),
        use_container_width=True,
    )

    for name, r in res.items():
        st.markdown(f"---\n#### {name}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", r["Accuracy"])
        m2.metric("Precision", r["Precision"])
        m3.metric("Recall", r["Recall"])
        m4.metric("Macro F1", r["Macro_F1"])

        chart_col, tree_col = st.columns([1.2, 0.8])
        with chart_col:
            cm = r["confusion_matrix"]
            labels = sorted(pd.Series(r["y_test"]).astype(str).unique())
            fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Blues")
            fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=380, **PLOTLY_THEME)
            st.plotly_chart(fig_cm, use_container_width=True)
            report_df = pd.DataFrame(r["report"]).T
            st.dataframe(report_df, use_container_width=True)

        with tree_col:
            if name.startswith("Random Forest"):
                try:
                    payload = _download_and_unpickle(r["model_id"])
                except ApiError as exc:
                    st.caption(f"Tree view unavailable: {exc}")
                    payload = None
                if payload is not None and hasattr(payload["model"], "estimators_"):
                    st.markdown("##### Random Forest Tree View")
                    model = payload["model"]
                    tree_count = len(model.estimators_)
                    tree_idx = st.slider(f"Tree index for {name}", 0, tree_count - 1, 0, key=f"rf_tree_{name}")
                    plot_depth = st.slider(f"Tree plot depth for {name}", 1, 6, 3, key=f"rf_plot_depth_{name}")
                    fig_tree, ax = plt.subplots(figsize=(10, 5))
                    plot_tree(
                        model.estimators_[tree_idx],
                        feature_names=r["features"],
                        class_names=[str(c) for c in model.classes_],
                        filled=True,
                        max_depth=plot_depth,
                        fontsize=7,
                        ax=ax,
                    )
                    plt.tight_layout()
                    st.pyplot(fig_tree, use_container_width=True)
                    plt.close(fig_tree)

        try:
            model_bytes = api_client.download_model(r["model_id"])
        except ApiError as exc:
            st.caption(f"Model download unavailable: {exc}")
        else:
            st.download_button(
                f"⬇ Download {name} model",
                model_bytes,
                file_name=f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl",
                mime="application/octet-stream",
                key=f"download_cls_{name}",
            )
