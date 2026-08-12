from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from utils.app_state import active_test_df, active_train_df, init_state
from utils.data_input import render_data_sidebar
from utils.ui import PLOTLY_THEME, dataset_preview_metrics, render_metadata_panel, setup_page, show_dataset_block


setup_page("DatoScope")
init_state()
render_data_sidebar()

st.title("DatoScope")
st.caption("Exploratory Data Analysis")

if st.session_state.train_df is None:
    st.info("Load or generate a dataset first.")
    st.stop()

render_metadata_panel(st.session_state.data_meta, st.session_state.train_df, st.session_state.test_df)
dataset_preview_metrics(st.session_state.train_df, "Train")
show_dataset_block("Train Dataset", st.session_state.train_df)

if st.session_state.data_meta.get("source") == "generated":
    st.markdown("---")
    st.markdown("#### Generated Dataset Plot")
    generated_df = st.session_state.train_df.copy()
    generated_num_cols = generated_df.select_dtypes(include="number").columns.tolist()
    target_col = st.session_state.data_meta.get("target_column")
    color_col = None
    for candidate in [target_col, "label", "target"]:
        if candidate and candidate in generated_df.columns:
            color_col = candidate
            break

    if len(generated_num_cols) >= 2:
        plot_features = [col for col in generated_num_cols if col != color_col]
        if len(plot_features) >= 2:
            x_col, y_col = plot_features[:2]
        else:
            x_col, y_col = generated_num_cols[:2]
        fig_generated = px.scatter(
            generated_df,
            x=x_col,
            y=y_col,
            color=color_col,
            opacity=0.8,
            title=f"Generated data preview: {x_col} vs {y_col}",
        )
        fig_generated.update_traces(marker=dict(size=9, line=dict(width=0.5, color="#111827")))
        fig_generated.update_layout(height=430, **PLOTLY_THEME)
        st.plotly_chart(fig_generated, use_container_width=True)
        if color_col:
            st.caption(
                f"This preview uses the first two numeric features and colors points by `{color_col}` so you can quickly inspect the generated structure."
            )
        else:
            st.caption("This preview uses the first two numeric features so you can quickly inspect the generated structure.")
    elif len(generated_num_cols) == 1:
        fig_generated = px.histogram(
            generated_df,
            x=generated_num_cols[0],
            color=color_col,
            nbins=40,
            title=f"Generated data preview: distribution of {generated_num_cols[0]}",
        )
        fig_generated.update_layout(height=380, **PLOTLY_THEME)
        st.plotly_chart(fig_generated, use_container_width=True)
        st.caption("Only one numeric feature is available, so the preview is shown as a distribution plot.")
    else:
        st.info("A generated-data preview plot is unavailable because no numeric feature columns were found.")

if st.session_state.test_df is not None:
    st.markdown("---")
    dataset_preview_metrics(st.session_state.test_df, "Test")
    show_dataset_block("Test Dataset", st.session_state.test_df)

if st.session_state.clean_train_df is not None:
    st.markdown("---")
    st.markdown(
        """
        <div class="cleaned-preview-banner">
          <b>Cleaned Dataset Preview</b><br>
          Your processed train dataset is ready for EDA, supervised modeling, and clustering.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Cleaned Train Preview")
    report = st.session_state.clean_report_train or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows after cleaning", f"{report.get('rows_out', len(st.session_state.clean_train_df)):,}")
    c2.metric("Columns after cleaning", f"{report.get('cols_out', st.session_state.clean_train_df.shape[1])}")
    c3.metric("Actions taken", f"{len(report.get('actions', []))}")
    if report.get("actions"):
        with st.expander("Train cleaning log"):
            for action in report["actions"]:
                st.write(f"• {action}")
    st.dataframe(st.session_state.clean_train_df.head(200), use_container_width=True, height=260)
    st.download_button(
        "⬇ Download cleaned train CSV",
        st.session_state.clean_train_df.to_csv(index=False).encode(),
        file_name="datascope_clean_train.csv",
        mime="text/csv",
        key="eda_download_clean_train",
    )

    if st.session_state.clean_test_df is not None:
        st.markdown("#### Cleaned Test Preview")
        test_report = st.session_state.clean_report_test or {}
        if test_report.get("actions"):
            with st.expander("Test cleaning log"):
                for action in test_report["actions"]:
                    st.write(f"• {action}")
        st.dataframe(st.session_state.clean_test_df.head(200), use_container_width=True, height=240)
        st.download_button(
            "⬇ Download cleaned test CSV",
            st.session_state.clean_test_df.to_csv(index=False).encode(),
            file_name="datascope_clean_test.csv",
            mime="text/csv",
            key="eda_download_clean_test",
        )

st.markdown("---")
st.markdown("#### Missing Values (Train)")
miss = st.session_state.train_df.isnull().mean() * 100
miss = miss[miss > 0].sort_values(ascending=False)
if miss.empty:
    st.success("No missing values in the train dataset")
else:
    fig_miss = px.bar(
        x=miss.index,
        y=miss.values,
        labels={"x": "Column", "y": "Missing %"},
        color=miss.values,
        color_continuous_scale="plasma",
    )
    fig_miss.update_layout(coloraxis_showscale=False, height=320, **PLOTLY_THEME)
    st.plotly_chart(fig_miss, use_container_width=True)

st.markdown("---")
st.markdown("### EDA Analysis")

dataset_choice = st.selectbox("Analyze dataset", ["Train"] + (["Test"] if active_test_df() is not None else []))
df_eda = active_train_df() if dataset_choice == "Train" else active_test_df()
if df_eda is None:
    st.warning("Selected dataset is unavailable.")
    st.stop()

num_cols = df_eda.select_dtypes(include="number").columns.tolist()
if not num_cols:
    st.warning("No numeric columns found.")
    st.stop()

st.markdown("#### Summary Statistics")
st.caption(
    "Interpretation guide: skewness near 0 is fairly symmetric, between 0.5 and 1.0 in magnitude suggests moderate skew, and above 1.0 suggests strong skew. "
    "For kurtosis, values near 0 are close to normal-tail behavior, above 0 indicate heavier tails and more extreme values, while below 0 indicate lighter tails."
)
stats_df = df_eda[num_cols].describe().T
stats_df["skewness"] = df_eda[num_cols].skew().round(3)
stats_df["kurtosis"] = df_eda[num_cols].kurtosis().round(3)
st.dataframe(stats_df.style.background_gradient(cmap="Blues", subset=["mean", "std"]), use_container_width=True)

st.markdown("---")
st.markdown("#### Feature Distributions")
sel_cols = st.multiselect("Choose features", num_cols, default=num_cols[: min(4, len(num_cols))])
if sel_cols:
    n = len(sel_cols)
    nc = min(3, n)
    nr = (n + nc - 1) // nc
    fig = make_subplots(rows=nr, cols=nc, subplot_titles=sel_cols, vertical_spacing=0.12)
    for idx, col in enumerate(sel_cols):
        row, col_idx = divmod(idx, nc)
        fig.add_trace(
            go.Histogram(x=df_eda[col].dropna(), nbinsx=40, name=col, marker_color="#00d4ff", opacity=0.75),
            row + 1,
            col_idx + 1,
        )
    fig.update_layout(showlegend=False, height=300 * nr, **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("#### Box Plots")
st.caption("Box plots summarize each selected feature using the median, quartiles, spread, and potential outliers. Outlier percentages below use the standard IQR rule, so you can quickly see how much of each feature lies beyond the whisker range.")
box_cols = st.multiselect(
    "Choose features for box plots",
    num_cols,
    default=num_cols[: min(4, len(num_cols))],
    key="box_cols",
)
if box_cols:
    outlier_stats = []
    bn = len(box_cols)
    bcols = min(2, bn)
    brows = (bn + bcols - 1) // bcols
    subplot_titles = []
    for feature in box_cols:
        series = df_eda[feature].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = round((outlier_count / len(series)) * 100, 2) if len(series) else 0.0
        outlier_stats.append(
            {
                "Feature": feature,
                "Outlier %": outlier_pct,
                "Outlier Count": outlier_count,
                "Lower Bound": round(lower, 3),
                "Upper Bound": round(upper, 3),
            }
        )
        subplot_titles.append(f"{feature}<br><sup>{outlier_pct}% outliers</sup>")
    box_fig = make_subplots(rows=brows, cols=bcols, subplot_titles=subplot_titles, vertical_spacing=0.14)
    for idx, feature in enumerate(box_cols):
        row, col_idx = divmod(idx, bcols)
        box_fig.add_trace(
            go.Box(
                y=df_eda[feature].dropna(),
                name=feature,
                boxmean="sd",
                marker_color="#7c3aed",
                fillcolor="rgba(124,58,237,0.28)",
                line=dict(color="#00d4ff", width=1.5),
                showlegend=False,
            ),
            row + 1,
            col_idx + 1,
        )
    box_fig.update_layout(height=320 * brows, **PLOTLY_THEME)
    st.plotly_chart(box_fig, use_container_width=True)
    st.dataframe(
        pd.DataFrame(outlier_stats).sort_values("Outlier %", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )

st.markdown("---")
st.markdown("#### Q-Q Plots")
st.caption("Q-Q plots compare each feature against a normal distribution. Strong bends away from the diagonal suggest skew, heavy tails, or non-normal behavior.")
qq_cols = st.multiselect("Choose features for Q-Q plots", num_cols, default=num_cols[: min(2, len(num_cols))], key="qq_cols")
if qq_cols:
    qn = len(qq_cols)
    qcols = min(2, qn)
    qrows = (qn + qcols - 1) // qcols
    qq_fig = make_subplots(rows=qrows, cols=qcols, subplot_titles=qq_cols)
    for idx, col in enumerate(qq_cols):
        row, col_idx = divmod(idx, qcols)
        series = df_eda[col].dropna()
        osm, osr = stats.probplot(series, dist="norm", fit=False)
        slope, intercept, _ = stats.probplot(series, dist="norm", fit=True)[1]
        qq_fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color="#00d4ff", size=5), showlegend=False), row + 1, col_idx + 1)
        x_line = np.array([min(osm), max(osm)])
        y_line = slope * x_line + intercept
        qq_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="#f59e0b", dash="dash"), showlegend=False), row + 1, col_idx + 1)
    qq_fig.update_layout(height=320 * qrows, **PLOTLY_THEME)
    st.plotly_chart(qq_fig, use_container_width=True)

st.markdown("---")
st.markdown("#### Correlation Matrix")
if len(num_cols) >= 2:
    corr = df_eda[num_cols].corr()
    fig_c = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig_c.update_layout(height=500, **PLOTLY_THEME)
    st.plotly_chart(fig_c, use_container_width=True)

    corr_flat = corr.mask(np.triu(np.ones(corr.shape, dtype=bool))).stack().reset_index()
    corr_flat.columns = ["Feature A", "Feature B", "Correlation"]
    corr_flat["abs"] = corr_flat["Correlation"].abs()
    top = corr_flat.nlargest(10, "abs").drop(columns="abs")
    with st.expander("Top 10 correlated pairs"):
        st.dataframe(top.reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.markdown("#### Scatter Plot")
if len(num_cols) >= 2:
    col1, col2 = st.columns(2)
    x_col = col1.selectbox("X axis", num_cols, index=0, key=f"eda_x_{dataset_choice}")
    y_col = col2.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1), key=f"eda_y_{dataset_choice}")
    color_opts = ["— none —"] + df_eda.columns.tolist()
    color_col = st.selectbox("Color by", color_opts, key=f"eda_color_{dataset_choice}")
    fig_s = px.scatter(df_eda, x=x_col, y=y_col, color=None if color_col == "— none —" else color_col, opacity=0.65, trendline="ols")
    fig_s.update_layout(height=420, **PLOTLY_THEME)
    st.plotly_chart(fig_s, use_container_width=True)

st.markdown("---")
st.markdown("#### Feature Variance Ranking")
st.caption("Variance shows how much a feature spreads out. Higher variance means values are more spread and may carry more separating power, while very low variance often means the feature changes very little and may add limited signal.")
var = df_eda[num_cols].var().sort_values(ascending=False)
var_df = var.reset_index()
var_df.columns = ["Feature", "Variance"]
var_df["Interpretation"] = np.where(
    var_df["Variance"] < max(var_df["Variance"].max() * 0.05, 0.01),
    "Low spread",
    np.where(var_df["Variance"] > var_df["Variance"].median(), "High spread", "Moderate spread"),
)
fig_v = px.bar(
    var_df,
    x="Feature",
    y="Variance",
    color="Interpretation",
    text=var_df["Variance"].round(3),
    color_discrete_map={"High spread": "#00d4ff", "Moderate spread": "#7c3aed", "Low spread": "#f59e0b"},
)
fig_v.update_traces(textposition="outside")
fig_v.update_layout(height=360, yaxis_title="Variance (spread of values)", **PLOTLY_THEME)
st.plotly_chart(fig_v, use_container_width=True)
