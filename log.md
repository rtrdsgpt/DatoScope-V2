# DatoScope V2 — Build Log

Key decisions, mistakes/fixes, and progress, logged after each section of `todo.md`.

---

## Section 0 — Fork setup

**Progress:** [rtrdsgpt/DatoScope-V2](https://github.com/rtrdsgpt/DatoScope-V2) created and pushed
— public, MIT-licensed, single fresh initial commit.

**Decisions**
- Not a GitHub "fork" — cloned `tanmoyghosh704-lang/DatoScope` (branch `mehak-work`), dropped
  `.git` entirely, and re-initialized with a single fresh commit. User wanted no traceback to the
  original repo beyond a README credit line, so GitHub's fork lineage and the original commit
  history were both dropped (a deliberate choice — the alternative, keeping history but not using
  GitHub's Fork button, was offered and declined).
- Repo name `DatoScope-V2`, MIT license — user picked both from options over alternatives
  (keeping the short name `DatoScope`, or a different license).
- Public visibility — matches the todo's stated goal of this being a portfolio project for the
  Data Science CV.

**Cleanup done**
- Removed 4 committed `.DS_Store` files, added `.DS_Store` to `.gitignore`.
- Fixed `.devcontainer/devcontainer.json`: `openFiles`/`postAttachCommand` referenced a deleted
  `Preprocessing.py`; pointed at `app.py` (the real Streamlit entry point — `app.py` builds
  `st.navigation` over `pages/`, there's no separate `Preprocessing.py` page).
- Fixed README: removed the stale `Preprocessing.py` "official entry page" description, fixed the
  `Project Structure` tree (missing `data_loader.py`/`utils/io.py`, referenced deleted
  `Preprocessing.py`), replaced hardcoded `mehakgupta`-specific absolute paths in Setup/Run
  instructions with relative commands.
- Added `LICENSE` (MIT).

**Mistakes & fixes:** none of note — straightforward.

---

## Section 1 — ETL pipeline

**Progress:** `etl/` package (extract → transform → validate → load) plus an Airflow DAG, all
verified end-to-end against real services (MinIO, Postgres, Airflow via docker-compose) and real
data (Kaggle). Pushed to `main`.

**Decisions**
- **External Extract source: Kaggle**, via `kagglehub` (not the classic `kaggle` package) —
  `kagglehub` is Kaggle's current recommended library, simpler API
  (`kagglehub.dataset_download(handle)`), reads the same credential sources. Default dataset is
  `uciml/iris` (small, reliable, well-known — good default for a pipeline that's dataset-agnostic
  by design).
- **Data quality tool: Great Expectations** over Pandera — user's explicit choice, traded GE's
  heavier setup for its stronger "enterprise DE tool" name recognition. Used GE's ephemeral
  context (`gx.get_context(mode="ephemeral")`) with suites built in code, not YAML — avoids
  scaffolding a full GE project for a suite that's generated per-dataset at runtime anyway.
- **Raw zone: MinIO via docker-compose**, not a local-filesystem stand-in — user's explicit
  choice, so the "S3 raw zone" the todo describes is real (S3-compatible API via boto3) starting
  now, rather than deferred until section 5's Terraform/AWS work provisions a real bucket.
  `etl/storage.py`'s `ObjectStore` talks to MinIO locally and real S3 in prod via the same boto3
  client — only endpoint/credentials change.
- **Warehouse layout: one Postgres table per dataset**, not one shared wide table — dataset
  schemas vary too much (generated/uploaded/Kaggle datasets have completely different columns) for
  a single table to make sense. A shared `etl_runs` registry table indexes every run
  (dataset_name, run_id, table_name, source, row/col counts, loaded_at) so a future consumer
  (FastAPI in section 2) can discover what's in the warehouse without knowing table names.
- **XCom carries references, not DataFrames** — each Airflow task pushes only
  `{bucket, data_key, meta_key, run_id, dataset_name}`; the next task re-reads the actual data from
  the raw/processed zone by key. Directly reused from the Financial Anomaly Detection Using RAG
  project's DAG pattern (thin `PythonOperator`s around shared pipeline functions), per the todo's
  explicit instruction to reuse that structure rather than reinvent it.
- **Airflow image kept separate from the main app**: `requirements-etl.txt` (boto3, sqlalchemy,
  psycopg2, great-expectations, kagglehub, pandas/numpy/sklearn/scipy) is a deliberately smaller
  subset of `requirements.txt` — the Airflow container has no reason to carry Streamlit/Plotly/
  Matplotlib.
- Split `parse_uploaded_bytes` out of `utils/preprocessing.py` as a Streamlit-free pure function
  (the old `load_uploaded_file` called `st.error`/`st.stop()` directly, coupling file parsing to a
  Streamlit runtime). `load_uploaded_file` is now a thin wrapper around it. Necessary so
  `etl/extract.py` — and the Streamlit-less Airflow image — can reuse the exact same parsing logic
  the app uses, not a duplicate copy.

**Mistakes & fixes**
1. **Kaggle credential format.** User supplied a `KGAT_...` token with no username. Traced through
   `kagglehub`'s source (`kagglehub/config.py` → `kagglesdk.kaggle_env.get_access_token_from_env`)
   to confirm this is Kaggle's newer unified-token format, read from a single `KAGGLE_API_TOKEN`
   env var (not the legacy `KAGGLE_USERNAME`/`KAGGLE_KEY` pair). Verified by actually pulling
   `uciml/iris` before building anything else on top of it.
2. **SQLAlchemy version split between two environments.** `etl/load.py` originally pinned
   `sqlalchemy>=2.0.0` in both `requirements.txt` (host) and `requirements-etl.txt` (Airflow
   image), for "consistency." Building the Airflow image failed: Airflow 2.9.3 hard-pins
   `SQLAlchemy<2.0` (and `flask-appbuilder<1.5`). Fix attempt #1 — install
   `requirements-etl.txt` against Airflow's own constraints file
   (`constraints-2.9.3/constraints-3.11.txt`), which is the documented way to add packages to an
   Airflow image without breaking its pins — still failed because `requirements-etl.txt` itself
   requested `sqlalchemy>=2.0.0`, directly contradicting the constraint. Fix attempt #2 — pinned
   `sqlalchemy<2.0` in *both* requirement files for "consistency," which broke the **host** venv
   instead: pandas 3.0 (resolved from `requirements.txt`'s unpinned `pandas>=2.0.0`) requires
   SQLAlchemy 2.0+ for `to_sql`; downgrading it there raised
   `AttributeError: 'Connection' object has no attribute 'cursor'`. Root cause understood: the two
   environments have different pandas versions (host resolves pandas 3.0.5; Airflow's base image
   ships pandas 2.1.4, pinned by its own constraints file) and each needs the SQLAlchemy major that
   matches *its own* pandas — they don't need to match each other. Final fix — host
   `requirements.txt` kept `sqlalchemy>=2.0.0`; only `requirements-etl.txt` (the Airflow image) was
   pinned to `sqlalchemy>=1.4.36,<2.0`. Also changed `etl/load.py`'s import from
   `from sqlalchemy import Engine` (2.0-only top-level export) to
   `from sqlalchemy.engine import Engine` (valid on both 1.4 and 2.0), so the same source file
   works unmodified in both environments.
3. **YAML folded-scalar bug in docker-compose's `airflow-init` command.** Copied the reference
   project's multi-line `command: >` block verbatim (progressively-indented continuation lines).
   YAML's folding rule only folds lines at the *same* indentation into spaces — more-indented lines
   are preserved as literal, separately-newlined lines. Because each continuation line was indented
   one step deeper than the last, bash received `airflow users create --username admin --password
   admin` and `--firstname Admin ...` as two separate commands, so the second failed with
   `--firstname: command not found`. Fixed by flattening the whole thing to a single-line `command:
   -c "..."` string — no folding ambiguity possible. (The reference project likely carries the same
   latent bug; it just may not have been re-run since being written.)
4. **`ModuleNotFoundError: No module named 'etl'` inside the Airflow container.** The DAG file
   lives in `/opt/airflow/dags` (a mounted volume); `etl/`, `utils/`, `data_loader.py` were copied
   to `/opt/airflow/` by the Dockerfile. Airflow puts the dags folder on `sys.path` but not its
   parent, so `from etl.pipeline import ...` failed on import. Fixed by adding `PYTHONPATH:
   /opt/airflow` to the shared `airflow-env` anchor in docker-compose.
5. **`ModuleNotFoundError: No module named 'streamlit'` inside the Airflow container**, immediately
   after fixing #4 — `utils/__init__.py` eagerly imports `utils.preprocessing`, which imported
   `streamlit` at module level purely for the `load_uploaded_file` error path. This is what forced
   the `parse_uploaded_bytes` split described above (Decisions) — the pure parser has no Streamlit
   dependency, and the streamlit import moved to be local inside `load_uploaded_file` only.
6. **`docker compose run --rm airflow-init bash -c "..."` → `cannot execute binary file`.** The
   service's `entrypoint` is already `/bin/bash`; passing `bash -c "..."` as the run command made
   Airflow's bash invoke `bash <args>`, i.e. treat the literal string `"bash"` as a script filename
   to execute (a binary, not a script) rather than the `-c` flag. Fixed by dropping the leading
   `bash` and passing `-c "..."` directly as the override command.

**Verification performed** (not just "written," actually run):
- Full pipeline (`run_pipeline`) executed in-process against live MinIO + Postgres for all three
  Extract sources (kaggle/`uciml/iris`, generated, uploaded) — confirmed rows in the warehouse
  table and both raw/processed objects in MinIO.
- `validate_dataframe` tested against both a clean fixture (0 failures) and a deliberately bad one
  (missing required column + 60% nulls) — confirmed it raises `DataQualityError` with the correct
  failed-expectation list rather than silently passing.
- Airflow image built clean (no pip resolver conflicts) and `airflow dags test datoscope_etl_dag
  <date>` ran all 4 tasks (`extract → transform → validate → load`) to `state=success` against the
  same live MinIO/Postgres services; confirmed a fresh `iris`/`kaggle` row appeared in `etl_runs`
  afterward, proving the DAG-driven run actually wrote through, not just parsed.

**Housekeeping:** `airflow/logs/` added to `.gitignore` (runtime artifacts from the `dags test`
run); `.env`/`.env.example` set up so `KAGGLE_API_TOKEN` never risks being committed.

---

## Section 2 — API layer

**Progress:** `api/` (FastAPI) built and verified end-to-end against live MinIO/Postgres — every
router (datasets, eda, modeling, clustering, comparison) exercised with real HTTP requests over
real generated/uploaded/Kaggle data, not just written and assumed correct.

**Decisions**
- **Scoped to "backend first, verified standalone" rather than a full Streamlit rewire in the same
  pass** — user's explicit choice over doing both at once. `utils/modeling.py`'s results contain
  trained sklearn objects, confusion matrices, and raw prediction arrays (none directly
  JSON-serializable), and the 4 Streamlit pages (~1300 lines) are tightly built around
  `st.session_state`. Rewiring all of that to be a pure HTTP client in the same pass as designing
  the backend's contracts was assessed as high regression-risk for currently-working UI; todo.md
  section 2 is now split into two checkboxes (backend done, Streamlit-as-client deferred) rather
  than one, to track this honestly instead of overclaiming.
- **Ingestion split into two calls, mirroring the ETL stages**: `POST /datasets/{generate,upload,
  kaggle}` does extract only (lands raw, returns a `run_id`); `POST /datasets/{name}/clean` does
  transform + validate + load. This lets a client preview a raw extract under different cleaning
  parameters before committing one to the warehouse — the same "adjust options, see a live
  estimate, then commit" shape the current Streamlit sidebar already has (`estimate_outlier_
  removal` before the "Clean & Preprocess" button) — rather than forcing cleaning params to be
  chosen blind at extract time.
- **One warehouse table per dataset stays the read path too** (not just write, from section 1) —
  added `get_dataset`/`list_datasets`/`list_runs`/`get_run` to `etl/load.py` rather than a new
  module, since it already owns the engine/table-naming logic and reading back is the natural
  complement to writing.
- **Raw-run lookup by dataset_name alone** (`etl/extract.find_raw_run`) — the API's `/clean`
  endpoint takes a `dataset_name` without requiring the caller to know which of the three sources
  (generated/uploaded/kaggle) produced it. Implemented by listing S3 keys under all three
  `raw/<source>/<dataset_name>/` prefixes and taking the lexicographically-max run_id — safe
  because `run_id` is a sortable UTC timestamp string by construction (section 1).
- **EDA endpoints return data, not plots** — `pages/1_EDA.py`'s Plotly rendering stays a Streamlit
  concern; the API returns the underlying numbers (histogram bin edges/counts, box-plot quartiles,
  Q-Q plot points, correlation matrices) so a future Streamlit rewire (or any other client) can
  plot them however it wants.
- **Comparison logic extracted into `utils/comparison.py`**, ported from `pages/4_Comparison.py`'s
  `score_regression_models`/`score_clustering_models` but rewritten to operate on plain metric
  dicts (not DataFrames holding model objects) — reused by `api/routers/comparison.py`. Page 4
  itself was deliberately left untouched (still has its own inline copy) since it's out of scope
  until the Streamlit rewire pass; noted here so the duplication is intentional, not forgotten.
- **In-memory model registry for `/modeling/download/{model_id}`** — trained sklearn models are
  kept in a module-level dict keyed by UUID, not persisted. Acceptable for a local dev API (mirrors
  today's session-scoped download button); doesn't survive an API restart or scale across workers,
  which is fine at this project's current stage and noted directly in the 404 error message rather
  than silently failing.
- **Metric key renaming**: `utils/modeling.py` uses display-oriented keys like `"R²"`, `"CV R²"`,
  `"Overfit Gap"` (unicode/spaces, fine for a Streamlit dataframe column, awkward as a JSON/API
  contract). `api/serialization.py` renames these to plain identifiers (`R2`, `CV_R2`,
  `Overfit_Gap`, ...) on the way out, matching what `utils/comparison.py` expects — so a
  `/modeling/.../regression` response can be fed directly into `/comparison/regression` with no
  client-side reshaping.

**Mistakes & fixes:** none blocking this time — the section 1 SQLAlchemy/pandas version lesson
(host env needs 2.0+ for pandas 3.0's `to_sql`; only the Airflow image is pinned to 1.4) carried
forward cleanly since the API runs in the host env. One real test-harness slip caught immediately:
first curl against `/datasets/generate` used `dataset_type: "Two Moons"` with `task_type:
"Classification"` — `"Two Moons"` is a *clustering* dataset type (`CLASSIFICATION_DATASETS` are
`"Linearly Separable"`/`"Overlapping Classes"`/`"Imbalanced Classes"`); `utils/generators.py`
correctly raised `KeyError` rather than silently returning the wrong data. Not a code bug — retried
with a valid dataset type.

**Verification performed** (real requests against live MinIO/Postgres, not just written):
- `POST /datasets/generate` → `POST /datasets/{name}/clean` → confirmed report/validation/load
  summary, for classification, regression, and clustering generated datasets.
- `POST /datasets/upload` (real multipart file) → clean → `GET /datasets/{name}/data` round-trip.
- `POST /datasets/kaggle` (`uciml/iris`, real Kaggle pull) → extract confirmed.
- All 7 EDA endpoints (summary, missing, distributions, boxplot, qq, correlation, variance) called
  against a real cleaned dataset and checked for correct shapes/values.
- `POST /modeling/.../regression` and `.../classification` — checked metrics, then fed a real
  response straight into `POST /comparison/regression` / `/comparison/classification` and
  confirmed the winner matched hand-checked expectations (highest Macro F1 for classification;
  highest generalization_score for regression).
- `POST /clustering/{name}` with a real `ground_truth_col` — confirmed FM Score/Rand Index appear,
  projection returns 300 points matching input row count, and `/comparison/clustering` correctly
  broke a 2-way `metric_wins` tie using Silhouette (matching `pages/4_Comparison.py`'s tiebreak
  order).
- `GET /modeling/download/{model_id}` — downloaded a real `.pkl`, unpickled it outside the API
  process, and called `.predict()` on the restored `KNeighborsClassifier` to confirm it's a
  genuinely fitted model, not just a serialized stub.
- Error paths: 404 for an unknown dataset (`/eda/...`, `/datasets/.../clean`), 422 with the full
  Great Expectations failure report for a `/clean` request whose `label_col` doesn't exist in the
  data.
