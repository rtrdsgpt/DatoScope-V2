"""
Shared sidebar data-input controls for the DatoScope app.

Talks to the DatoScope API (utils.api_client) instead of calling
utils.generators/utils.preprocessing directly: Generate/Upload land raw data
via the ETL pipeline's extract stage, Clean & Preprocess runs transform +
validate + load. The one local computation kept client-side is the live
"about to remove N rows" outlier estimate — a cheap, side-effect-free
preview over data already fetched from the API, not a second source of
truth for the dataset itself.
"""

from __future__ import annotations

import hashlib
import re

import streamlit as st

from utils import api_client
from utils.api_client import ApiError
from utils.app_state import set_clean_data, set_raw_data
from utils.generators import CLASSIFICATION_DATASETS, CLUSTERING_DATASETS, REGRESSION_DATASETS
from utils.preprocessing import estimate_outlier_removal
from utils.ui import render_sidebar_brand


def _uploaded_signature(uploaded) -> str:
    if uploaded is None:
        return ""
    raw = uploaded.getvalue()
    return hashlib.md5(raw).hexdigest()


def _sanitize_dataset_name(name: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", name)
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", stem).strip("_").lower()
    return slug or "dataset"


def render_data_sidebar() -> None:
    with st.sidebar:
        render_sidebar_brand()

        data_mode = st.radio(
            "Data source",
            ["Generate Dataset", "Upload Single File", "Upload Train/Test"],
            index=["Generate Dataset", "Upload Single File", "Upload Train/Test"].index(st.session_state.data_source_mode)
            if st.session_state.data_source_mode in ["Generate Dataset", "Upload Single File", "Upload Train/Test"]
            else 1,
        )

        if data_mode == "Generate Dataset":
            task_type = st.selectbox("Task type", ["Regression", "Classification", "Clustering"])
            dataset_options = (
                REGRESSION_DATASETS if task_type == "Regression"
                else CLASSIFICATION_DATASETS if task_type == "Classification"
                else CLUSTERING_DATASETS
            )
            dataset_type = st.selectbox("Dataset type", dataset_options)
            n_samples = st.slider("Samples", 100, 5000, 500, step=50)
            noise = st.slider("Noise", 0.01, 2.0, 0.15, step=0.01)
            n_clusters = st.slider("Clusters / arms", 2, 10, 3) if task_type == "Clustering" else 3
            min_features = 2 if task_type == "Clustering" else 1
            default_features = 20 if dataset_type == "High-Dimensional" else max(min_features, 6)
            n_features = st.slider("Number of features", min_features, 50, default_features)
            n_informative = (
                st.slider("Informative features", 2, n_features, min(5, n_features))
                if task_type == "Classification" or dataset_type == "High-Dimensional"
                else 5
            )
            target_name = st.text_input(
                "Target column name",
                value="label" if task_type == "Clustering" else "target",
                disabled=task_type == "Clustering",
            )
            create_split = st.checkbox("Create train/test split now", value=task_type != "Clustering")
            generated_test_pct = st.slider("Generated test split %", 10, 40, 20) if create_split and task_type != "Clustering" else 20
            random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

            if st.button("🧪 Generate Dataset", use_container_width=True):
                with st.spinner("Generating dataset…"):
                    try:
                        resp = api_client.generate_dataset(
                            task_type=task_type,
                            dataset_type=dataset_type,
                            n_samples=n_samples,
                            noise=noise,
                            n_clusters=n_clusters,
                            random_seed=random_seed,
                            n_features=n_features,
                            n_informative=n_informative,
                            target_name=target_name if task_type != "Clustering" else "label",
                            create_split=create_split and task_type != "Clustering",
                            test_split_pct=generated_test_pct,
                        )
                        train_df = api_client.get_raw(resp["dataset_name"], resp["run_id"])
                        test_df, test_dataset_name, test_run_id = None, "", ""
                        if resp.get("test"):
                            test_dataset_name = resp["test"]["dataset_name"]
                            test_run_id = resp["test"]["run_id"]
                            test_df = api_client.get_raw(test_dataset_name, test_run_id)
                    except ApiError as exc:
                        st.error(f"Generate failed: {exc}")
                    else:
                        meta = {**resp["generator_meta"], "active_ml_task": task_type}
                        set_raw_data(
                            dataset_name=resp["dataset_name"],
                            run_id=resp["run_id"],
                            train_df=train_df,
                            test_dataset_name=test_dataset_name,
                            test_run_id=test_run_id,
                            test_df=test_df,
                            train_filename=f"{task_type.lower()}_{dataset_type.lower().replace(' ', '_')}.csv",
                            source_mode="Generate Dataset",
                            metadata=meta,
                        )
                        st.session_state.single_upload_signature = ""
                        st.session_state.train_upload_signature = ""
                        st.session_state.test_upload_signature = ""
                        st.success(f"Generated {dataset_type} dataset")

        elif data_mode == "Upload Single File":
            uploaded = st.file_uploader("Upload dataset", type=["csv", "xls", "xlsx", "zip", "data"], key="single_upload")
            if uploaded:
                upload_sig = _uploaded_signature(uploaded)
                if upload_sig != st.session_state.single_upload_signature:
                    with st.spinner("Uploading…"):
                        try:
                            dataset_name = _sanitize_dataset_name(uploaded.name)
                            resp = api_client.upload_dataset(uploaded.name, uploaded.getvalue(), dataset_name)
                            train_df = api_client.get_raw(resp["dataset_name"], resp["run_id"])
                        except ApiError as exc:
                            st.error(f"Upload failed: {exc}")
                        else:
                            set_raw_data(
                                dataset_name=resp["dataset_name"],
                                run_id=resp["run_id"],
                                train_df=train_df,
                                train_filename=uploaded.name,
                                source_mode="Upload Single File",
                                metadata={"source": "uploaded", "split_method": "Auto split from uploaded train file", "dataset_type": "User upload"},
                            )
                            st.session_state.data_meta["active_ml_task"] = "Auto"
                            st.session_state.single_upload_signature = upload_sig
                            st.session_state.train_upload_signature = ""
                            st.session_state.test_upload_signature = ""
                            st.success(f"Loaded {uploaded.name}")
                            st.caption(f"{train_df.shape[0]:,} rows × {train_df.shape[1]} cols")

        else:
            train_upload = st.file_uploader("Train file", type=["csv", "xls", "xlsx", "zip", "data"], key="train_upload")
            test_upload = st.file_uploader("Test file (optional)", type=["csv", "xls", "xlsx", "zip", "data"], key="test_upload")
            if train_upload:
                train_sig = _uploaded_signature(train_upload)
                test_sig = _uploaded_signature(test_upload)
                if train_sig != st.session_state.train_upload_signature or test_sig != st.session_state.test_upload_signature:
                    with st.spinner("Uploading…"):
                        try:
                            dataset_name = _sanitize_dataset_name(train_upload.name)
                            train_resp = api_client.upload_dataset(train_upload.name, train_upload.getvalue(), dataset_name)
                            train_df = api_client.get_raw(train_resp["dataset_name"], train_resp["run_id"])
                            test_df, test_dataset_name, test_run_id = None, "", ""
                            if test_upload:
                                test_resp = api_client.upload_dataset(test_upload.name, test_upload.getvalue(), f"{dataset_name}__test")
                                test_df = api_client.get_raw(test_resp["dataset_name"], test_resp["run_id"])
                                test_dataset_name, test_run_id = test_resp["dataset_name"], test_resp["run_id"]
                        except ApiError as exc:
                            st.error(f"Upload failed: {exc}")
                        else:
                            if test_df is not None and list(train_df.columns) != list(test_df.columns):
                                st.error("Train/test column mismatch. Upload files with the same schema.")
                            else:
                                set_raw_data(
                                    dataset_name=train_resp["dataset_name"],
                                    run_id=train_resp["run_id"],
                                    train_df=train_df,
                                    test_dataset_name=test_dataset_name,
                                    test_run_id=test_run_id,
                                    test_df=test_df,
                                    train_filename=train_upload.name,
                                    test_filename=test_upload.name if test_upload else "",
                                    source_mode="Upload Train/Test",
                                    metadata={
                                        "source": "uploaded",
                                        "split_method": "Uploaded test file" if test_df is not None else "Auto split from uploaded train file",
                                        "dataset_type": "User upload",
                                    },
                                )
                                st.session_state.data_meta["active_ml_task"] = "Auto"
                                st.session_state.single_upload_signature = ""
                                st.session_state.train_upload_signature = train_sig
                                st.session_state.test_upload_signature = test_sig
                                st.success("Train/test files loaded" if test_df is not None else "Train file loaded")

        st.divider()

        if st.session_state.train_df is not None:
            current_task = st.session_state.data_meta.get("active_ml_task", "Auto")
            task_options = ["Auto", "Regression", "Classification", "Clustering"]
            task_index = task_options.index(current_task) if current_task in task_options else 0
            st.session_state.data_meta["active_ml_task"] = st.selectbox("Selected ML task", task_options, index=task_index)
            st.markdown("#### Preprocessing")
            missing_strat = st.selectbox("Missing values", ["mean", "median", "mode", "drop"])
            outlier_meth = st.selectbox("Outlier method", ["IQR", "Z-Score", "None"])
            scale_meth = st.selectbox("Scaler", ["Standard", "MinMax", "Robust"])
            encode_categoricals = st.checkbox("Encode categorical variables", value=False)
            remove_dupes = st.checkbox("Remove duplicates", value=True)
            train_cols = st.session_state.train_df.columns.tolist()
            label_col = st.selectbox("Label / target column (optional — keeps it unscaled)", ["— none —"] + train_cols)
            label_col = None if label_col == "— none —" else label_col
            st.session_state.data_meta["target_column"] = label_col or "Not selected yet"

            categorical_cols = [
                col for col in st.session_state.train_df.select_dtypes(exclude="number").columns.tolist() if col != label_col
            ]
            categorical_encoding = "One-Hot"
            categorical_encoding_map: dict[str, str] = {}
            if encode_categoricals:
                if categorical_cols:
                    categorical_encoding = st.selectbox("Default categorical encoding", ["One-Hot", "Label"])
                    st.caption("Choose the encoding method for each categorical feature below.")
                    for col_name in categorical_cols:
                        categorical_encoding_map[col_name] = st.selectbox(
                            f"Encoding for {col_name}",
                            ["One-Hot", "Label"],
                            index=0 if categorical_encoding == "One-Hot" else 1,
                            key=f"encoding_{col_name}",
                        )
                else:
                    st.caption("No categorical feature columns are available for encoding in the current train dataset.")

            if outlier_meth == "None":
                st.caption("Outlier removal is off, so no rows will be removed by the outlier step.")
            else:
                train_removed, train_base = estimate_outlier_removal(
                    st.session_state.train_df,
                    missing_strategy=missing_strat,
                    outlier_method=outlier_meth,
                    remove_dupes=remove_dupes,
                    label_col=label_col,
                )
                train_pct = (train_removed / train_base * 100) if train_base else 0.0
                st.caption(
                    f"{outlier_meth} will remove about {train_removed} train row(s) "
                    f"({train_pct:.2f}% of {train_base} rows after missing-value handling and duplicate removal)."
                )
                if st.session_state.test_df is not None:
                    test_removed, test_base = estimate_outlier_removal(
                        st.session_state.test_df,
                        missing_strategy=missing_strat,
                        outlier_method=outlier_meth,
                        remove_dupes=remove_dupes,
                        label_col=label_col if label_col in st.session_state.test_df.columns else None,
                    )
                    test_pct = (test_removed / test_base * 100) if test_base else 0.0
                    st.caption(
                        f"On the test dataset, the same setting will remove about {test_removed} row(s) "
                        f"({test_pct:.2f}% of {test_base})."
                    )

            if st.button("⚙️ Clean & Preprocess", use_container_width=True):
                with st.spinner("Cleaning datasets…"):
                    clean_kwargs = dict(
                        missing_strategy=missing_strat,
                        outlier_method=outlier_meth if outlier_meth != "None" else "none",
                        scale_method=scale_meth,
                        remove_dupes=remove_dupes,
                        label_col=label_col,
                        encode_categoricals=encode_categoricals,
                        categorical_encoding=categorical_encoding,
                        categorical_encoding_map=categorical_encoding_map,
                    )
                    try:
                        train_resp = api_client.clean_dataset(st.session_state.dataset_name, run_id=st.session_state.run_id, **clean_kwargs)
                        train_clean = api_client.get_data(st.session_state.dataset_name, train_resp["run_id"])
                        test_resp, test_clean = None, None
                        if st.session_state.test_dataset_name:
                            test_label_col = label_col if label_col in (st.session_state.test_df.columns if st.session_state.test_df is not None else []) else None
                            test_resp = api_client.clean_dataset(
                                st.session_state.test_dataset_name,
                                run_id=st.session_state.test_run_id,
                                **{**clean_kwargs, "label_col": test_label_col},
                            )
                            test_clean = api_client.get_data(st.session_state.test_dataset_name, test_resp["run_id"])
                    except ApiError as exc:
                        st.error(f"Clean & Preprocess failed: {exc}")
                    else:
                        set_clean_data(
                            processed_run_id=train_resp["run_id"],
                            clean_train_df=train_clean,
                            report_train=train_resp["report"],
                            test_processed_run_id=test_resp["run_id"] if test_resp else "",
                            clean_test_df=test_clean,
                            report_test=test_resp["report"] if test_resp else None,
                        )
                st.success("Dataset preprocessing complete")

        st.divider()
        st.caption("DatoScope: An Interactive Approach to Data Visualization and Machine Learning")
        st.caption("Created by: Aritra , Tanmoy , Mehak")
