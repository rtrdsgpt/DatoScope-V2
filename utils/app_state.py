"""
Shared Streamlit session-state helpers for DatoScope.

The app is an API client: session state holds dataset_name/run_id
references to warehouse-backed data (fetched via utils.api_client) rather
than being the source of truth itself. train_df/clean_train_df etc. are a
local cache of what the API returned, used for rendering.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


DEFAULT_STATE = dict(
    theme="light",
    data_source_mode="Upload Single File",
    single_upload_signature="",
    train_upload_signature="",
    test_upload_signature="",
    dataset_name="",
    run_id="",  # raw extract run_id
    processed_run_id="",  # warehouse run_id, set after Clean & Preprocess
    test_dataset_name="",
    test_run_id="",
    test_processed_run_id="",
    raw_df=None,
    train_df=None,
    test_df=None,
    clean_df=None,
    clean_train_df=None,
    clean_test_df=None,
    filename="",
    train_filename="",
    test_filename="",
    clean_report=None,
    clean_report_train=None,
    clean_report_test=None,
    data_meta={},
    validation_messages=[],
    regression_target="",
    classification_target="",
    reg_results={},
    cls_results={},
    cluster_results={},
)


def init_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_model_results() -> None:
    st.session_state.reg_results = {}
    st.session_state.cls_results = {}
    st.session_state.cluster_results = {}


def set_raw_data(
    *,
    dataset_name: str,
    run_id: str,
    train_df: pd.DataFrame,
    test_dataset_name: str = "",
    test_run_id: str = "",
    test_df: pd.DataFrame | None = None,
    train_filename: str,
    test_filename: str = "",
    source_mode: str,
    metadata: dict | None = None,
) -> None:
    """Record a fresh extract (post generate/upload/kaggle, pre-clean)."""
    st.session_state.data_source_mode = source_mode
    st.session_state.dataset_name = dataset_name
    st.session_state.run_id = run_id
    st.session_state.test_dataset_name = test_dataset_name
    st.session_state.test_run_id = test_run_id
    st.session_state.processed_run_id = ""
    st.session_state.test_processed_run_id = ""
    st.session_state.raw_df = train_df.copy()
    st.session_state.train_df = train_df.copy()
    st.session_state.test_df = test_df.copy() if test_df is not None else None
    st.session_state.clean_df = None
    st.session_state.clean_train_df = None
    st.session_state.clean_test_df = None
    st.session_state.clean_report = None
    st.session_state.clean_report_train = None
    st.session_state.clean_report_test = None
    st.session_state.filename = train_filename
    st.session_state.train_filename = train_filename
    st.session_state.test_filename = test_filename
    st.session_state.data_meta = metadata or {}
    st.session_state.validation_messages = []
    reset_model_results()


def set_clean_data(
    *,
    processed_run_id: str,
    clean_train_df: pd.DataFrame,
    report_train: dict,
    test_processed_run_id: str = "",
    clean_test_df: pd.DataFrame | None = None,
    report_test: dict | None = None,
) -> None:
    """Record the result of Clean & Preprocess (post /datasets/{name}/clean)."""
    st.session_state.processed_run_id = processed_run_id
    st.session_state.test_processed_run_id = test_processed_run_id
    st.session_state.clean_train_df = clean_train_df
    st.session_state.clean_df = clean_train_df
    st.session_state.clean_test_df = clean_test_df
    st.session_state.clean_report = report_train
    st.session_state.clean_report_train = report_train
    st.session_state.clean_report_test = report_test
    reset_model_results()


def active_train_df(use_clean: bool = True) -> pd.DataFrame | None:
    if use_clean and st.session_state.clean_train_df is not None:
        return st.session_state.clean_train_df
    return st.session_state.train_df


def active_test_df(use_clean: bool = True) -> pd.DataFrame | None:
    if use_clean and st.session_state.clean_test_df is not None:
        return st.session_state.clean_test_df
    return st.session_state.test_df


def active_dataset_name() -> str | None:
    """Dataset name to pass to the API for EDA/modeling/clustering — needs Clean & Preprocess run first."""
    if st.session_state.processed_run_id:
        return st.session_state.dataset_name
    return None


def add_validation(message: str) -> None:
    messages = st.session_state.get("validation_messages", [])
    if message not in messages:
        messages.append(message)
        st.session_state.validation_messages = messages
