"""
Shared sidebar data-input controls for the DatoScope app.
"""

from __future__ import annotations

import hashlib

import streamlit as st
from sklearn.model_selection import train_test_split

from utils.app_state import reset_model_results, set_data
from utils.generators import CLASSIFICATION_DATASETS, CLUSTERING_DATASETS, REGRESSION_DATASETS, generate_dataset
from utils.preprocessing import clean_datasets, estimate_outlier_removal, load_uploaded_file
from utils.ui import render_sidebar_brand


def _uploaded_signature(uploaded) -> str:
    if uploaded is None:
        return ""
    raw = uploaded.getvalue()
    return hashlib.md5(raw).hexdigest()


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
                df_gen, meta = generate_dataset(
                    task_type,
                    dataset_type,
                    n_samples=n_samples,
                    noise=noise,
                    n_clusters=n_clusters,
                    random_seed=random_seed,
                    n_features=n_features,
                    n_informative=n_informative,
                    target_name=target_name if task_type != "Clustering" else "label",
                )
                active_target = meta.get("target_column", target_name if task_type != "Clustering" else "label")
                train_df = df_gen
                test_df = None
                if create_split and task_type != "Clustering":
                    stratify = train_df[active_target] if task_type == "Classification" else None
                    train_df, test_df = train_test_split(
                        df_gen,
                        test_size=generated_test_pct / 100,
                        random_state=random_seed,
                        stratify=stratify,
                    )
                    train_df = train_df.reset_index(drop=True)
                    test_df = test_df.reset_index(drop=True)
                    meta["split_method"] = f"Generated split ({generated_test_pct}% test)"
                else:
                    meta["split_method"] = "Single generated dataset"
                meta["active_ml_task"] = task_type
                set_data(
                    train_df=train_df,
                    test_df=test_df,
                    raw_df=df_gen,
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
                    df_raw = load_uploaded_file(uploaded)
                    set_data(
                        train_df=df_raw,
                        test_df=None,
                        raw_df=df_raw,
                        train_filename=uploaded.name,
                        source_mode="Upload Single File",
                        metadata={"source": "uploaded", "split_method": "Auto split from uploaded train file", "dataset_type": "User upload"},
                    )
                    st.session_state.data_meta["active_ml_task"] = "Auto"
                    st.session_state.single_upload_signature = upload_sig
                    st.session_state.train_upload_signature = ""
                    st.session_state.test_upload_signature = ""
                    st.success(f"Loaded {uploaded.name}")
                    st.caption(f"{df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

        else:
            train_upload = st.file_uploader("Train file", type=["csv", "xls", "xlsx", "zip", "data"], key="train_upload")
            test_upload = st.file_uploader("Test file (optional)", type=["csv", "xls", "xlsx", "zip", "data"], key="test_upload")
            if train_upload:
                train_sig = _uploaded_signature(train_upload)
                test_sig = _uploaded_signature(test_upload)
                if train_sig != st.session_state.train_upload_signature or test_sig != st.session_state.test_upload_signature:
                    train_df = load_uploaded_file(train_upload)
                    test_df = load_uploaded_file(test_upload) if test_upload else None
                    if test_df is not None and list(train_df.columns) != list(test_df.columns):
                        st.error("Train/test column mismatch. Upload files with the same schema.")
                    else:
                        set_data(
                            train_df=train_df,
                            test_df=test_df,
                            raw_df=train_df,
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
                    train_clean, test_clean, train_report, test_report = clean_datasets(
                        st.session_state.train_df,
                        st.session_state.test_df,
                        missing_strategy=missing_strat,
                        outlier_method=outlier_meth if outlier_meth != "None" else "none",
                        scale_method=scale_meth,
                        remove_dupes=remove_dupes,
                        label_col=label_col,
                        encode_categoricals=encode_categoricals,
                        categorical_encoding=categorical_encoding,
                        categorical_encoding_map=categorical_encoding_map,
                    )
                    st.session_state.clean_df = train_clean
                    st.session_state.clean_train_df = train_clean
                    st.session_state.clean_test_df = test_clean
                    st.session_state.clean_report = train_report
                    st.session_state.clean_report_train = train_report
                    st.session_state.clean_report_test = test_report
                    reset_model_results()
                st.success("Dataset preprocessing complete")

        st.divider()
        st.caption("DatoScope: An Interactive Approach to Data Visualization and Machine Learning")
        st.caption("Created by: Aritra , Tanmoy , Mehak")
