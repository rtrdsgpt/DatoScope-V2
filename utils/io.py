"""
utils/io.py
Shared helpers used across all DatoScope scripts.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader import load_any_dataset

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
OUTPUTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
PLOTS_DIR    = os.path.join(OUTPUTS_DIR, "plots")

def ensure_dirs():
    for d in [DATASETS_DIR, OUTPUTS_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

def save_dataset(df: pd.DataFrame, filename: str) -> str:
    ensure_dirs()
    path = os.path.join(DATASETS_DIR, filename)
    df.to_csv(path, index=False)
    return path

def load_dataset(filename: str) -> pd.DataFrame:
    path = os.path.join(DATASETS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}\nRun 01_generate_data.py first.")
    return load_any_dataset(path)

def save_output(df: pd.DataFrame, filename: str) -> str:
    ensure_dirs()
    path = os.path.join(OUTPUTS_DIR, filename)
    df.to_csv(path, index=False)
    return path

def load_output(filename: str) -> pd.DataFrame:
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Output not found: {path}\nRun 02_clean_data.py first.")
    return pd.read_csv(path)

def list_datasets(directory: str = "datasets") -> list:
    base = DATASETS_DIR if directory == "datasets" else OUTPUTS_DIR
    valid_exts = (".csv", ".xls", ".xlsx", ".data")
    return sorted(f for f in os.listdir(base) if f.endswith(valid_exts) and not f.startswith("_"))

def build_registry(directory: str = "datasets") -> pd.DataFrame:
    """Scan a directory and produce a summary registry of all datasets."""
    base = DATASETS_DIR if directory == "datasets" else OUTPUTS_DIR
    records = []
    valid_exts = (".csv", ".xls", ".xlsx", ".data")
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(valid_exts) or fname.startswith("_"):
            continue
        try:
            df = load_any_dataset(os.path.join(base, fname)) if directory == "datasets" else pd.read_csv(os.path.join(base, fname))
        except Exception:
            continue
        
        task = "clustering" if "clustering" in fname else ("regression" if "regression" in fname else "unknown")
        records.append({
            "filename"  : fname,
            "task"      : task,
            "n_rows"    : df.shape[0],
            "n_cols"    : df.shape[1],
            "columns"   : list(df.columns),
            "has_label" : "label"  in df.columns,
            "has_target": "target" in df.columns,
        })
    return pd.DataFrame(records)
