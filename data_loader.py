"""
data_loader.py
DatoScope — Unified Dataset Loader

Features
--------
* Supports .csv, .xls, .xlsx, .data, and zipped variants (.csv.zip)
* Dtype optimisation  — downcasts numeric columns to save memory
* Chunked CSV reading — streams large files in configurable chunks
* File-level caching  — returns cached DataFrames for repeated calls
* Smart Excel header  — detects multi-row headers without double-reading
* Informative errors  — clear messages on missing files / bad formats
"""

from __future__ import annotations

import os
import zipfile
import io
import functools
import logging
from pathlib import Path
from typing import Optional, Iterator

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE: int = 50_000       # rows per chunk for large CSV reads
LARGE_FILE_THRESHOLD: int = 50 * 1024 * 1024  # 50 MB — use chunked loading
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".csv", ".xls", ".xlsx", ".data", ".zip"}
)


# ── In-memory cache ───────────────────────────────────────────────────────────
# Keyed on (resolved_path, use_cache=True).  Call clear_cache() to reset.
_CACHE: dict[str, pd.DataFrame] = {}


def clear_cache() -> None:
    """Remove all cached DataFrames."""
    _CACHE.clear()
    log.debug("DataLoader cache cleared.")


# ── Public API ────────────────────────────────────────────────────────────────

def load_any_dataset(
    filepath: str | os.PathLike,
    *,
    use_cache: bool = True,
    optimise_dtypes: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    encoding: str = "utf-8",
    **kwargs,
) -> pd.DataFrame:
    """
    Load a dataset from *filepath* regardless of its format.

    Parameters
    ----------
    filepath        : path to the file (.csv, .xls, .xlsx, .data, or .csv.zip)
    use_cache       : return a cached copy when the same path is requested again
    optimise_dtypes : downcast int/float columns to the smallest safe dtype
    chunk_size      : rows per chunk when streaming large CSV files
    encoding        : character encoding for CSV / .data files
    **kwargs        : extra arguments forwarded to the underlying pandas reader

    Returns
    -------
    pd.DataFrame
    """
    path = Path(filepath).resolve()
    cache_key = str(path)

    if use_cache and cache_key in _CACHE:
        log.debug("Cache hit: %s", path.name)
        return _CACHE[cache_key].copy()

    _assert_exists(path)
    log.info("Loading: %s  (%s)", path.name, _human_size(path.stat().st_size))

    # Dispatch by extension
    ext = _effective_extension(path)

    if ext == ".csv":
        df = _load_csv(path, chunk_size=chunk_size, encoding=encoding, **kwargs)
    elif ext in {".xls", ".xlsx"}:
        df = _load_excel(path, **kwargs)
    elif ext == ".data":
        df = _load_data_file(path, encoding=encoding, **kwargs)
    elif ext == ".zip":
        df = _load_zipped_csv(path, chunk_size=chunk_size, encoding=encoding, **kwargs)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}' for: {path}\n"
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if optimise_dtypes:
        df = _optimise_dtypes(df)

    if use_cache:
        _CACHE[cache_key] = df.copy()

    log.info(
        "Loaded %s — %d rows × %d cols  (%.1f MB in memory)",
        path.name, len(df), len(df.columns), df.memory_usage(deep=True).sum() / 1e6,
    )
    return df


def stream_csv_chunks(
    filepath: str | os.PathLike,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    encoding: str = "utf-8",
    **kwargs,
) -> Iterator[pd.DataFrame]:
    """
    Yield successive *chunk_size*-row DataFrames from a CSV file.

    Useful for datasets too large to hold in memory all at once.

    Example
    -------
    >>> for chunk in stream_csv_chunks("deliveries.csv"):
    ...     process(chunk)
    """
    path = Path(filepath).resolve()
    _assert_exists(path)

    reader = pd.read_csv(
        path,
        chunksize=chunk_size,
        encoding=encoding,
        on_bad_lines="warn",
        **kwargs,
    )
    for chunk in reader:
        yield chunk


# ── Private helpers ───────────────────────────────────────────────────────────

def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Run 01_generate_data.py first, or check your datasets/ directory."
        )


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _effective_extension(path: Path) -> str:
    """
    Return the 'logical' extension.

    For files like  data.csv.zip  → ".zip" (handled as zipped CSV).
    For plain       data.csv      → ".csv".
    """
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".zip":
        return ".zip"
    return suffixes[-1] if suffixes else ""


# ── CSV ───────────────────────────────────────────────────────────────────────

def _load_csv(
    path: Path,
    chunk_size: int,
    encoding: str,
    **kwargs,
) -> pd.DataFrame:
    """Load a CSV, streaming through chunks when the file is large."""
    file_size = path.stat().st_size

    # Small file — read all at once (faster)
    if file_size < LARGE_FILE_THRESHOLD:
        return pd.read_csv(
            path,
            encoding=encoding,
            on_bad_lines="warn",
            low_memory=False,
            **kwargs,
        )

    # Large file — stream chunks then concatenate
    log.info(
        "Large file detected (%s) — reading in chunks of %d rows",
        _human_size(file_size), chunk_size,
    )
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        chunksize=chunk_size,
        encoding=encoding,
        on_bad_lines="warn",
        **kwargs,
    ):
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


# ── Zipped CSV ────────────────────────────────────────────────────────────────

def _load_zipped_csv(
    path: Path,
    chunk_size: int,
    encoding: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Extract the first CSV from a .zip archive and load it.

    Works for files like  datasets/deliveries.csv.zip.
    """
    with zipfile.ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(
                f"No .csv file found inside {path.name}. "
                f"Contents: {zf.namelist()}"
            )
        if len(csv_names) > 1:
            log.warning(
                "Multiple CSVs in %s — loading the first one: %s",
                path.name, csv_names[0],
            )
        raw = zf.read(csv_names[0])

    return _load_csv(
        path=path,           # used only for logging; actual data is in `raw`
        chunk_size=chunk_size,
        encoding=encoding,
        **{k: v for k, v in kwargs.items()},
        # Override path-based read with in-memory bytes
        **{"filepath_or_buffer": io.BytesIO(raw)} if True else {},
    )


# Simpler, cleaner implementation for zip (avoids **kwargs clash)
def _load_zipped_csv(  # type: ignore[no-redef]  # noqa: F811
    path: Path,
    chunk_size: int,
    encoding: str,
    **kwargs,
) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(
                f"No .csv file found inside {path.name}. "
                f"Contents: {zf.namelist()}"
            )
        if len(csv_names) > 1:
            log.warning(
                "Multiple CSVs in %s — loading: %s", path.name, csv_names[0]
            )
        raw_bytes = io.BytesIO(zf.read(csv_names[0]))

    file_size = len(raw_bytes.getvalue())
    if file_size < LARGE_FILE_THRESHOLD:
        return pd.read_csv(raw_bytes, encoding=encoding, low_memory=False, **kwargs)

    log.info("Large zipped CSV (%s) — streaming chunks", _human_size(file_size))
    chunks = [
        chunk
        for chunk in pd.read_csv(
            raw_bytes, chunksize=chunk_size, encoding=encoding, **kwargs
        )
    ]
    return pd.concat(chunks, ignore_index=True)


# ── Excel ─────────────────────────────────────────────────────────────────────

def _load_excel(path: Path, **kwargs) -> pd.DataFrame:
    """
    Load an Excel file.

    Detects multi-row headers with a single read by inspecting the first
    row's dtype pattern rather than reading the file twice.
    """
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"

    # Read a small preview (first 5 rows) to decide on the header row
    preview = pd.read_excel(
        path,
        engine=engine,
        nrows=5,
        header=0,
        **{k: v for k, v in kwargs.items() if k not in ("header", "nrows")},
    )

    # Heuristic: if every column in the preview is object-typed, the first
    # row is likely a secondary header — try header=1 using the same preview
    header_row = 0
    if len(preview.columns) > 0 and all(preview.dtypes == object):
        preview_h1 = pd.read_excel(
            path, engine=engine, nrows=5, header=1,
            **{k: v for k, v in kwargs.items() if k not in ("header", "nrows")},
        )
        if len(preview_h1.select_dtypes(include="number").columns) > 0:
            header_row = 1
            log.debug("Multi-row header detected — using header=1")

    return pd.read_excel(path, engine=engine, header=header_row, **{
        k: v for k, v in kwargs.items() if k not in ("header",)
    })


# ── .data files ───────────────────────────────────────────────────────────────

def _load_data_file(path: Path, encoding: str, **kwargs) -> pd.DataFrame:
    """
    Load a headerless .data file (e.g. drug_consumption.data).

    Tries comma-separated first, then whitespace-separated.
    Column names default to feat_0, feat_1, …
    """
    try:
        df = pd.read_csv(
            path, header=None, encoding=encoding, sep=",",
            on_bad_lines="warn", **kwargs,
        )
        if df.shape[1] == 1:
            raise ValueError("Only one column — probably not comma-separated")
    except (ValueError, pd.errors.ParserError):
        log.debug("%s appears whitespace-delimited — retrying", path.name)
        df = pd.read_csv(
            path, header=None, encoding=encoding, sep=r"\s+",
            on_bad_lines="warn", engine="python", **kwargs,
        )

    df.columns = [f"feat_{c}" for c in df.columns]
    return df


# ── Dtype optimisation ────────────────────────────────────────────────────────

def _optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest safe dtype.

    int64  → int8 / int16 / int32  (whichever fits)
    float64 → float32              (unless precision is critical)

    Skips non-numeric columns silently.
    """
    original_bytes = df.memory_usage(deep=True).sum()

    for col in df.select_dtypes(include=["integer"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        for dtype in (np.int8, np.int16, np.int32):
            info = np.iinfo(dtype)
            if info.min <= col_min and col_max <= info.max:
                df[col] = df[col].astype(dtype)
                break

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # Convert low-cardinality object columns to category
    for col in df.select_dtypes(include=["object"]).columns:
        n_unique = df[col].nunique(dropna=False)
        if n_unique / max(len(df), 1) < 0.5:          # < 50 % unique values
            df[col] = df[col].astype("category")

    optimised_bytes = df.memory_usage(deep=True).sum()
    if original_bytes > 0:
        saving_pct = 100 * (1 - optimised_bytes / original_bytes)
        log.debug(
            "Dtype optimisation: %.1f MB → %.1f MB  (%.0f %% saving)",
            original_bytes / 1e6, optimised_bytes / 1e6, saving_pct,
        )

    return df