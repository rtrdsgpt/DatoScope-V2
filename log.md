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

---

## Section 2 (continued) — Streamlit becomes an API client

**Progress:** the deferred half of section 2 — `utils/api_client.py` plus a rewrite of the sidebar
(`utils/data_input.py`, `utils/app_state.py`) and all four pages to call the API over HTTP instead
of `utils.generators`/`utils.preprocessing`/`utils.modeling` directly. Verified in an actual
browser (Playwright, headless Chromium — no `chromium-cli`/Node available in this environment, so
installed `playwright` into the project venv and drove it with a Python script instead), not just
started and assumed working.

**Decisions**
- **A real bug was found before any UI code was touched**: `etl/transform.transform_raw` reused
  the raw extract's `run_id` for the processed/warehouse output too. The new interactive "tweak
  cleaning params, click Clean & Preprocess again" flow this rewire adds calls `/clean` repeatedly
  against the *same* raw extract — which would have silently written duplicate rows into the
  warehouse under one `run_id` every time. Fixed by having `transform_raw` mint its own fresh
  `run_id` (via a new shared `etl/storage.new_run_id()`, also adopted by `extract.py`) and carry
  the raw extract's id forward as `source_run_id` for lineage instead. Verified directly (not just
  inferred): called `/clean` twice against one raw extract with different parameters and confirmed
  two distinct, correctly-row-counted warehouse runs rather than one doubled one. This is exactly
  the kind of bug that only surfaces when you build the actual caller, not just the endpoint —
  logged here because it would have been easy to ship silently.
- **The section 2 API design had missed the "separate test dataset" flow** — `RegressionRequest`/
  `ClassificationRequest` only ever passed `df_test=None` to `utils.modeling`, so the "Upload
  Train/Test" and "Generate with a train/test split" flows (real, existing app features) had no
  API equivalent. Added `test_dataset_name`/`test_run_id` to both request schemas and threaded
  them through to `run_regression_models`/`run_classification_models`'s existing `df_test`
  parameter — no new modeling logic needed, just exposing what was already there. Verified via the
  browser test: training against a split-generated dataset correctly reports `split_method:
  "Uploaded test file"` rather than an auto-split.
- **`/datasets/generate` now supports `create_split`/`test_split_pct` server-side** — splits with
  `sklearn.train_test_split` (same call the sidebar used to make locally) and extracts *both*
  halves as separate raw datasets (`<name>` and `<name>__test`) in one request, so lineage stays
  "generated" for both rather than the alternative (generate whole, fetch it back, split locally,
  re-upload each half as if it were a user upload).
- **Added `GET /datasets/{name}/raw`** (reads the raw zone directly, mirroring the existing
  `/data` warehouse endpoint) — needed because the EDA page's pre-clean views (raw dataset
  preview, the generated-data scatter, the "Missing Values (Train)" chart) need actual row-level
  data before anything has been committed to the warehouse, and the sidebar's live "about to
  remove N rows" outlier estimate needs the raw dataframe in hand to compute against.
- **EDA stats: API when cleaned, local fallback when not** — `GET /eda/...` only ever reads
  warehouse (processed) data, but the app has always let users explore *raw* data before deciding
  cleaning parameters (a deliberate, existing UX choice — auto-cleaning on extract would remove
  that). So `pages/1_EDA.py` calls the API once a dataset has a `processed_run_id` (the normal,
  expected path after Clean & Preprocess) and falls back to the exact same local pandas/scipy
  computation as before when it doesn't. Both paths produce identical numbers; only the "where it's
  computed" differs. This dual-path is why `pages/1_EDA.py` stayed large rather than shrinking.
- **Clustering diagnostics that need the raw scaled matrix (dendrogram, elbow curve) stayed
  client-side**, reading a `X` field added to the clustering API response (the same scaled
  DataFrame `utils.modeling.run_clustering_models` already returns internally) rather than adding
  narrow one-off endpoints for each diagnostic view. The *primary* clustering computation (fit +
  metrics + 2D projection) is fully server-side; these two supplementary views are page-specific
  and operate directly on data already fetched alongside the main result, the same way EDA's
  Plotly rendering stays client-side over API-computed numbers.
- **Random Forest tree view required fetching the model back**: `plot_tree` needs the actual
  fitted `RandomForestClassifier.estimators_`, which the API doesn't return in the training
  response (by design — that's what `/modeling/download/{model_id}` is for). `pages/2_Supervised_
  Modeling.py` downloads and unpickles the model via that endpoint, cached client-side with
  `@st.cache_data` keyed on `model_id` so the interactive tree-index/depth sliders (which trigger
  a Streamlit rerun on every change) don't re-download the model on each interaction.
- **Comparison page now calls `/comparison/*`** instead of its own inline `score_regression_
  models`/`score_clustering_models` copies — the duplication flagged as deliberate-for-now in the
  previous log entry is gone; both the API and the page use `utils/comparison.py`.

**Mistakes & fixes**
- Playwright/browser test harness: `chromium-cli` (the tool this environment's `run` skill
  documents) isn't installed and neither is Node/npm, so it was installed as `playwright` (Python)
  into the project venv instead, with Chromium downloaded via `playwright install chromium`. Not a
  project dependency — not added to `requirements.txt`, purely a one-off verification tool for
  this session.
- Several Playwright selector attempts against Streamlit's newer react-aria combobox widgets
  timed out intermittently (`get_by_role("option", ...)` after a `.click()` to open the dropdown —
  the floating listbox would sometimes close again before the click landed, likely racing a
  Streamlit rerender). Switched to a more robust `click()` → `fill()` → `Enter` pattern
  (type-ahead + keyboard select) for all dropdown interactions, which was reliable across every
  subsequent test run.

**Verification performed** (real browser, real API, real warehouse — screenshots inspected, not
just "no exception raised"):
- App loads with zero console errors.
- Sidebar: generated a Classification dataset with an 80/20 split (400/100 rows, confirmed in the
  Dataset Metadata panel), set a label column, ran Clean & Preprocess — outlier-removal estimate
  captions rendered correctly for both train and test before committing.
- EDA page: confirmed the "Stats computed via the API from the warehouse (`dataset_name`, run
  `...`)" caption appears, then visually inspected summary statistics, feature-distribution
  histograms (server-computed bin edges/counts rendered as `go.Bar`), box plots (rendered from
  precomputed quartiles, including the mean±std diamond marker), Q-Q plots, and the correlation
  heatmap — all correct.
- Supervised Modeling: trained Logistic Regression + Random Forest against a *separate uploaded
  test dataset* — confirmed "Evaluation split: Uploaded test file" (proving the new
  `test_dataset_name` wiring), confusion matrix, classification report, and the Random Forest Tree
  View rendering an actual decision tree from a model downloaded and unpickled from the API at
  runtime.
- Comparison page: correctly selected Random Forest as the winner (highest Macro F1) with the
  right supporting numbers in the "why this model won" text.
- Clustering: generated a dedicated clustering dataset, ran K-Means/DBSCAN/Hierarchical, confirmed
  ground-truth metrics (FM Score, Rand Index) from the auto-detected `label` column, the PCA
  scatter projection, cluster-size bars, and — reading the API's returned scaled matrix — a correct
  dendrogram.
- Zero console/page errors across the entire flow (generate → clean → EDA → model → compare →
  cluster).

---

## Section 3 — Testing

**Progress:** `tests/` — 107 tests total, all passing: 98 pure-function unit tests (no external
services, ~3s) plus 9 `@pytest.mark.integration` tests against the real docker-compose warehouse,
verified passing against live services (not just written).

**Decisions**
- **`moto` mocks S3 for the ETL unit tests** (`test_etl_extract.py`, `test_etl_transform.py`)
  instead of requiring live MinIO. Hit and worked around a real limitation: moto only intercepts
  the default AWS endpoint pattern, not an arbitrary custom `endpoint_url` like MinIO's — so
  `ObjectStore(Settings(s3_endpoint_url=None, ...))` is used specifically under `mock_aws()`,
  letting boto3 hit the (moto-intercepted) default AWS endpoint instead of trying to reach a real
  `localhost:9000`. This is a `moto`-specific test-setup detail, not a change to `etl/storage.py`
  itself.
- **`etl/load.py` and `etl/pipeline.py` are integration-only** (`@pytest.mark.integration`,
  skipped automatically if Postgres/MinIO aren't reachable) rather than mocked — Postgres-specific
  SQL (`TIMESTAMPTZ`, `ON CONFLICT`) doesn't have a good lightweight substitute (sqlite isn't a
  faithful stand-in), and the value of testing the real warehouse round-trip outweighs the cost of
  requiring `docker compose up -d minio warehouse` for that subset. Matches the verification bar
  from sections 1–2: actually run against live services, don't just assert the code "should" work.
- **Integration tests use randomly-suffixed `pytest_<uuid>` dataset names** so repeated/parallel
  runs don't collide in the shared warehouse, and the `warehouse_engine` fixture drops any
  `ds_pytest_*` tables and `etl_runs` rows it created in teardown — confirmed empty
  (`SELECT ... WHERE dataset_name LIKE 'pytest_%'` → 0 rows) after a full run, so the test suite
  doesn't leave debris in a database also used for manual dev/demo work.
- **`etl/validate.py` (Great Expectations) needed no infrastructure at all** — its ephemeral
  context runs fully in-process — so the data-quality test layer todo.md explicitly asks for
  (`tests/test_etl_validate.py`) is a plain unit test file: missing required columns, >max-null
  columns, empty tables, and out-of-bounds values (a synthetic `age` column with -5/150) each
  confirmed to raise `DataQualityError` carrying the specific failed-expectation type, not a bare
  assertion that *something* failed.
- **UI-layer code (`utils/api_client.py`, `utils/app_state.py`, `utils/data_input.py`,
  `utils/ui.py`) is intentionally 0% covered by this suite** — todo.md section 3 scopes testing to
  `preprocessing`/`modeling`/`generators`/ETL transform+validation, and that code was already
  exercised for real in section 2's Playwright browser pass; unit-testing Streamlit widget code
  needs heavy mocking for little signal beyond what the browser test already gave.

**Mistakes & fixes**
- **A real production bug, found by writing a test, not by reading the code**:
  `TestClassificationGenerator::test_imbalanced_weights_produce_skewed_classes` and the
  `generate_dataset` dispatcher's `"Imbalanced Classes"` case both failed with
  `AttributeError: 'tuple' object has no attribute 'copy'` — `utils/generators.py`'s
  `gen_classification` passes `weights` straight through to `sklearn.datasets.make_classification`,
  which internally calls `.copy()` on it; a tuple (what `gen_classification`'s own type hint and
  the dispatcher's `weights=(0.82, 0.18)` both use) doesn't have that method on the installed
  sklearn (1.9.0). This means **selecting "Imbalanced Classes" as a classification dataset type
  in the live app would have crashed** — a real, user-facing bug that existed before this session
  and had nothing to do with the ETL/API work. Fixed with a one-line cast
  (`weights=list(weights) if weights is not None else None`) in `gen_classification`, which fixes
  both the direct call and the dispatcher path since they share the same function.
- A handful of ordinary test-authoring mistakes, not product bugs, caught immediately by actually
  running the suite rather than assuming: `make_classification` fixture params summing to more
  features than requested (`n_informative + n_redundant + n_repeated > n_features`), and a
  `pd.testing.assert_frame_equal` that compared a freshly-`to_parquet`-round-tripped DataFrame
  (fresh `RangeIndex`) against the in-memory one (retains the original, non-contiguous index after
  `clean_dataframe` drops an outlier row) — fixed by `.reset_index(drop=True)` before comparing,
  since the index itself was never a meaningful part of what that test was checking.

**Verification performed:**
- `pytest -m "not integration"` — 98/98 pass in ~3s, no external services running.
- `pytest -m integration` with no services running — all 9 correctly *skip* (not error), proving
  the skip-gracefully behavior actually works rather than assuming it does.
- `docker compose up -d minio warehouse`, then `pytest -m integration` — all 9 pass for real
  against live Postgres/MinIO; confirmed zero `pytest_*` rows left in `etl_runs` afterward.
- Full combined run (`pytest tests/ --cov=utils --cov=etl`) with services up: 107/107 pass.
  Coverage on the modules todo.md section 3 asked for: `utils/preprocessing.py` 84%,
  `utils/modeling.py` 94%, `utils/generators.py` 99%, `utils/comparison.py` 96%,
  `etl/validate.py`/`etl/transform.py`/`etl/load.py` 100%, `etl/extract.py` 98%.

---

## Section 4 — MLOps

**Progress:** full dev-stack docker-compose (api + streamlit services, shared `Dockerfile`),
MLflow tracking + Model Registry wired into every training/comparison endpoint, DVC tracking
Production model artifacts against a MinIO remote, and a CI workflow (lint + pytest). CD
deliberately not built — explained below.

**Decisions**
- **CD deferred entirely, not stubbed** — user's explicit choice over the alternatives (stub the
  ECR/K8s steps as no-ops, or pause section 4 and go build sections 5–6 first). The CD job as
  specced (build → push to ECR → deploy to K8s) needs infrastructure that doesn't exist yet; a
  stub would be dead YAML nobody would remember to finish, and reordering sections would break the
  numbering/narrative of `todo.md`. Only the CI job (`.github/workflows/ci.yml`) exists for now.
- **MLflow backend: the existing warehouse Postgres + MinIO, not new infrastructure** — user's
  explicit choice over SQLite/local-filesystem. A new `mlflow` database in the same Postgres
  instance and a new `datoscope-mlflow` bucket in the same MinIO, both auto-created idempotently
  by `docker/mlflow_entrypoint.py` on container start (Postgres doesn't support `CREATE DATABASE
  IF NOT EXISTS`, so this does an explicit `pg_database` existence check first; MinIO doesn't
  auto-create buckets at all, so same idea via `head_bucket`/`create_bucket`).
- **DVC scoped to model artifacts only, not processed datasets** — user's explicit choice. DVC's
  usual job (versioning flat files in git) doesn't map cleanly onto this project's data, which
  lives in the Postgres warehouse (section 1), not flat files. Model artifacts are the one thing
  in the whole system that's still naturally file-shaped (a `.pkl`), so that's what DVC tracks.
- **Model export is a deliberate script + manual `dvc add`/`git commit`/`dvc push` sequence, not
  something the API does automatically on promotion** — a running web service mutating git history
  and pushing to a remote as a side effect of an HTTP request crosses a boundary that should stay
  a reviewable, human-triggered step (same reasoning as CD being deferred rather than automated
  prematurely). `scripts/05_export_production_model.py` only writes the file; README documents the
  rest of the sequence explicitly.
- **MLflow logging is best-effort everywhere it's wired in** (training endpoints, comparison's
  registration step) — model training or comparison already succeeded by the time MLflow is
  involved, so an unreachable MLflow server degrades to `mlflow_run_id: null` / `mlflow_error:
  "..."` in the response rather than failing the whole request. Mirrors the resilience posture
  already used for other non-critical-path failures in the API.
- **`pickle` serialization for MLflow model artifacts, not MLflow's `skops` default** — found
  during verification, not designed in upfront (see Mistakes below); matches what
  `utils.modeling.export_model_bytes` (the app's own model-download feature) already uses, so
  there's one serialization convention across the whole project instead of two.
- **`--allowed-hosts` explicitly set on the MLflow server** — found during verification (see
  Mistakes below); without it, MLflow 3.x's DNS-rebinding protection rejects every request whose
  Host header isn't `localhost`/a private IP, which silently includes the `api` container calling
  `http://mlflow:5000` from inside docker-compose.

**Mistakes & fixes**
1. **MLflow 3.x's new security middleware blocked cross-container requests.** First boot of the
   `mlflow` service looked healthy (`curl http://localhost:5001/` → 200), but a request with
   `Host: mlflow:5000` (simulating how the `api` container would actually reach it) got `403
   Invalid Host header - possible DNS rebinding attack detected`. MLflow 3.x added Host-header
   validation defaulting to localhost + private IPs only — a docker-compose service hostname isn't
   in that list. Fixed with `--allowed-hosts mlflow:5000,localhost:5000,localhost:5001,...`
   (exact `host:port` strings, not just hostnames — a first attempt with bare `mlflow,localhost`
   still failed the same way, since the default-replacement list does exact matching including
   port). Caught by deliberately testing with a spoofed Host header before building anything on
   top of the service, not by assuming a 200 on localhost meant it was actually reachable from
   where it needed to be reached from.
2. **MLflow's S3 artifact upload happens client-side, not proxied through the tracking server** —
   the first attempt at `mlflow.sklearn.log_model(...)` from a plain Python shell (with
   `MLFLOW_TRACKING_URI` set but no AWS/MinIO credentials in the environment) failed with
   `NoCredentialsError`. Params and metrics had already logged fine (visible in the MLflow UI),
   which is what made it clear only the artifact-upload path needed credentials — MLflow's S3
   artifact store is written to directly by whichever process calls `log_model`, not routed
   through the server. Fixed by setting `MLFLOW_S3_ENDPOINT_URL`/`AWS_ACCESS_KEY_ID`/
   `AWS_SECRET_ACCESS_KEY` (same MinIO credentials the ETL pipeline already uses) wherever
   `api/tracking.py` runs.
3. **KNN models failed to log with a real, non-obvious error**: `mlflow.sklearn.log_model` (skops
   serialization, MLflow's new default) rejected `KNeighborsClassifier` specifically —
   `Untrusted types found: ['sklearn.metrics._dist_metrics.EuclideanDistance64',
   'sklearn.neighbors._kd_tree.KDTree']`. Logistic Regression and Random Forest logged fine in the
   same request, which pinned this to skops' per-type trust allowlist rather than anything wrong
   with the training result itself (the best-effort error handling from decision #4 above caught
   it cleanly — training still succeeded, only that one model's `mlflow_run_id` came back `null`).
   Two fixes were possible: allowlist the specific internal types via `skops_trusted_types`, or
   switch to `pickle` serialization. Chose `pickle` — allowlisting is fragile whack-a-mole (every
   new sklearn estimator type used across regression/classification/clustering could hit the same
   wall) versus one serialization format that handles all of them uniformly, and it's what the
   rest of the app already uses for model export.
4. **`POST /models/{name}/promote` returned 500 instead of 404 for a nonexistent registered
   model.** `MlflowClient.get_latest_versions()` doesn't return an empty list for an unknown model
   name the way the code assumed — it raises `mlflow.exceptions.RestException:
   RESOURCE_DOES_NOT_EXIST`, uncaught, surfacing as a raw 500. Caught by deliberately testing the
   error path (`curl .../nonexistent_model/promote`) rather than only testing the happy path that
   had already worked. Fixed by catching `MlflowException` and re-raising as the router's own
   `ModelPromotionError` → 404.
5. **`ruff check .` with no config flagged 145 issues** using its full default-ish rule surface
   (bugbear, pep8-naming, bandit, datetimez, perflint, pylint, simplify, pyupgrade) — none of
   which is what "CI job = lint" was asking for on a codebase that predates this session. Scoped
   `pyproject.toml`'s `[tool.ruff.lint]` to `select = ["E", "F"]` (pycodestyle errors + pyflakes:
   real bugs, unused imports, undefined names) instead of gating CI on a large, unrelated style
   pass. Of the remaining 39 findings under that narrower selection: 32 were `utils/__init__.py`
   re-exporting submodule symbols for `from utils import X` (a legitimate, common pattern —
   per-file-ignored for `F401`), one was `df.dtypes == object` (idiomatic pandas, not a real
   type-comparison bug — ignored `E721` project-wide with a comment explaining why), and the
   remaining 6 (two genuinely unused imports in `data_loader.py`, two in test files) were real
   dead code, removed.

**Verification performed** (real docker-compose services, not just written and assumed correct):
- `docker compose up -d --build minio warehouse mlflow` — confirmed the `mlflow` service's
  entrypoint correctly creates its Postgres database and MinIO bucket on first boot (both
  previously absent) and that the tracking server responds on both `localhost:5001` (host) and a
  spoofed `mlflow:5000` Host header (simulating the `api` container).
- Full round trip for all three task types against the live API + MLflow: trained
  regression/classification/clustering models via the real endpoints, confirmed every model
  (including KNN, post-fix) got a real `mlflow_run_id`; ran `/comparison/*` and confirmed the
  actual winner (by the same logic tested in section 2) was registered under
  `<dataset_name>__<task>` and moved to Staging; called `/models/{name}/promote` and confirmed
  Staging → Production; listed versions and confirmed the stage transition stuck.
- DVC: `dvc add` a real exported Production model → `dvc push` → confirmed the object landed in
  MinIO directly via `boto3.list_objects_v2` (not just trusting DVC's own "1 file pushed" message)
  → deleted the local file → `dvc pull` → confirmed it came back byte-identical and still
  loadable/predictable as a real fitted model. Test artifact (`mlflow_test_cls`) cleaned up from
  both the local `models/` dir and the MinIO remote afterward so it doesn't ship in git history —
  same hygiene as section 3's `pytest_*` warehouse cleanup.
- `ruff check .` — clean pass after the config scoping + real dead-code removal above.
- Full `pytest` suite (98 unit tests) re-run after the ruff-driven import removals — still 98/98,
  confirming the "unused" imports really were unused and nothing broke.

---

## Section 7 — AI/agentic layer (done ahead of sections 5–6, at user's request)

**Progress:** grounded LLM co-pilot (EDA explanations + preprocessing recommendations, RAG over
installed sklearn/scipy docstrings, verified citations), an autonomous pipeline agent (goal →
extract/clean/EDA/train/compare → written report), and an MCP server exposing all of it — all
running on Groq's free tier. Verified end-to-end against live services, including finding and
fixing a real agent hallucination bug during that verification, not after.

**Decisions**
- **Reordered ahead of sections 5–6** — user's explicit request. No technical dependency requires
  AWS/Kubernetes first; the AI layer runs entirely against the local services sections 1–4 already
  built (warehouse, MinIO, the API, MLflow). CD from section 4 stays deferred until 5–6 exist.
- **LLM provider: Groq, not OpenAI or Anthropic** — went through a real decision chain, not a
  first-choice: OpenAI was the user's first pick, but the provided key had zero billing credits
  (confirmed via a real API call, not assumed from the key format). Considered Ollama next (already
  installed locally with a model pulled, zero cost) but its pulled model (`gemma4:e4b`) doesn't
  reliably support tool/function calling, which the autonomous agent genuinely needs — pulling a
  tool-calling-capable model was one option, but the user chose Groq's free tier instead. A
  `GROQ_API_KEY` was found already configured (as a comma-separated 5-key rotation pool) in a
  sibling project's `.env` at the user's direction, confirmed working with a real call before
  building anything on it.
- **Groq key rotation reused verbatim from the sibling Financial-Anomaly-Detection-Using-RAG
  project** (`agent/groq_client.py`, adapted from that project's `groq_key_rotation.py`) rather
  than reimplemented — same problem (a free-tier key's rate limit burns through mid-run), same
  proven solution, at the user's explicit direction ("No add rotation pool instead, Groq exhausts
  fast") after an initial attempt to just use the first key was rejected.
- **RAG corpus built from installed package docstrings** (`scripts/06_build_rag_corpus.py`,
  `inspect.getdoc()` over a curated list of the exact sklearn/scipy symbols DatoScope's
  preprocessing/modeling/EDA code uses) rather than authored prose — user's explicit choice,
  always in sync with the pinned dependency versions, no scraping or network fetch of arbitrary
  doc pages. 33 chunks, embedded locally with `sentence-transformers/all-MiniLM-L6-v2` (free, no
  API), retrieved via plain numpy cosine similarity — the corpus is a few dozen short chunks,
  nowhere near large enough to justify a real vector database.
- **Citation verification adapted from the same sibling project's `grounded_explainer.py`
  pattern**: the model must cite `[S1]`-style markers with an exact quoted substring, which is
  then deterministically checked against the actual retrieved source text — a real hallucination
  guard, not a prompt instruction the model may or may not follow. `unverified_citation_markers`
  in the response surfaces anything that didn't check out rather than silently trusting the model.
- **`agent/tools.py` wraps `utils/api_client.py` rather than reimplementing HTTP calls** — the
  same already-built, already-tested client from section 2's Streamlit rewire, reused for a third
  caller (the autonomous agent and the MCP server both import from `agent/tools.py`, which itself
  is just a thin trimming layer over `api_client`). One implementation, three callers total
  (Streamlit, the agent, MCP), not three copies of the same HTTP logic.
- **Training tools trim their output to only the metric fields `utils.comparison` actually reads**
  (dropping `y_test`/`y_pred`/confusion matrices/coefficients/cluster labels) — small enough for
  an LLM tool-call round trip, and this is what makes `compare_models` work by just forwarding a
  train_* tool's own output back in as its `results` argument, with no need for the agent to
  reconstruct or re-fetch anything.
- **Tool-calling JSON schemas are generated from `agent/tools.py`'s function signatures**
  (`agent/schema_gen.py`, via `typing.get_type_hints` + docstring parsing) rather than
  hand-maintained alongside the implementations — keeps the two from drifting apart by
  construction rather than by discipline.
- **MCP pinned to `mcp>=1.2.0,<2.0.0`** — `pip install mcp` resolved 2.0.0, which restructured the
  server API entirely (no more `mcp.server.fastmcp.FastMCP`, the class the sibling project's
  working MCP server is built on and the pattern this one follows). Rather than reverse-engineer
  the new 2.0 API from scratch, pinned to the well-documented 1.x line (1.29.0) that has the
  proven `FastMCP` class.

**Mistakes & fixes**
1. **`agent/schema_gen.py` produced silently wrong tool schemas on the first pass** — every
   `list[str]` parameter came back as `{"type": "string"}` and every `bool` as `{"type":
   "string"}` too, and tool descriptions truncated mid-sentence. Root cause: `agent/tools.py` uses
   `from __future__ import annotations`, which stringifies every annotation at definition time
   ("list[str]" as literal text, not a type object); `inspect.signature(func).parameters[name]
   .annotation` returns that raw string, and `typing.get_origin()`/dict-lookup on a string just
   silently falls through to the default. Fixed by resolving annotations through
   `typing.get_type_hints(func)` instead, which properly evaluates the postponed forward
   references back into real type objects. Caught by actually printing a generated schema and
   inspecting it before trusting it, not by assuming the generator worked because it didn't crash.
2. **Calling `api/routers/eda.py`'s route handlers directly (not through HTTP) crashed** —
   `api/routers/copilot.py`'s `_build_findings` called `summary(dataset_name, run_id)` as a plain
   Python function to build the co-pilot's context, and it crashed with `TypeError: 'Query' object
   is not iterable`. Root cause: `summary`'s `columns` parameter defaults to FastAPI's `Query(None)`
   marker, which is only resolved into a real value by FastAPI's request-handling machinery — call
   the function directly and `columns` is the raw `Query` object, not `None` or a list. Same bug
   existed in `agent/mcp_server.py`'s two copilot tools, which made the identical direct call.
   Fixed by extracting the actual computation out of `summary`/`missing`/`boxplot` into plain
   functions (`_summary_impl`, `_missing_impl`, `_boxplot_impl`) that take an already-fetched
   DataFrame and plain column list — no FastAPI markers anywhere in them — with the route handlers
   now thin wrappers that fetch the df and delegate. `copilot.py` and `mcp_server.py` both call the
   plain versions (the latter via `copilot.py`'s `_build_findings`, not duplicated).
3. **The citation verifier flagged every legitimate citation as unverified** on the first real
   grounded-answer test — the model's `source_text` was word-for-word correct but never
   substring-matched because docstrings wrap mid-sentence with embedded newlines
   ("...divided by the square of the\nvariance...") while the model naturally quotes the same text
   as flowing single-line prose (correctly — that's what an accurate quote of wrapped text looks
   like). Fixed by normalizing whitespace on both sides of the comparison before checking. Verified
   the fix didn't just make the check pass-everything by separately confirming it still correctly
   flags a genuinely fabricated citation.
4. **The autonomous agent hallucinated a complete, confident success report after every substantive
   tool call had actually failed.** First real end-to-end run: the model invented a placeholder
   `run_id="cleaning_run"` for `clean_dataset` (rather than reusing the real run_id
   `generate_dataset` had just returned, or omitting the optional parameter), which cascaded — every
   subsequent step (`clean_dataset`, `eda_summary`, `train_classification`, `compare_models`) failed
   with real, correctly-surfaced errors — and the model's final report ignored all of that and wrote
   a specific, plausible-looking success narrative (fabricated accuracy/precision/recall numbers
   for a model that was never trained). This is a serious problem for an "autonomous agent produces
   a written report" feature: a fabricated report is worse than an obvious failure, because it
   looks trustworthy. Three fixes, not one:
   - Rewrote the system prompt with explicit, direct rules: never invent an identifier (run_id,
     column name, etc.) — use only literal values from the goal or values verbatim from a previous
     tool result; most `run_id` parameters are optional and mean "use the latest" when omitted, so
     omit rather than inventing a placeholder; a tool result containing an `"error"` key means that
     step failed and must not be treated as a success or built upon; final-report numbers must come
     only from actual tool results, never asserted from memory.
   - Added a **deterministic integrity check independent of the model's own narrative**: the agent
     loop scans its own tool-call trace for any result containing `"error"` and returns
     `had_tool_errors: true/false` in the response alongside the free-text report — a caller (or a
     human) doesn't have to trust the model's self-assessment of whether the run actually worked.
   - `agent/tools.py`'s `compare_models` now validates its `results` argument is actually a dict of
     dicts and raises a clear, specific error otherwise, instead of crashing later inside
     `score_classification_models` with a cryptic `TypeError: string indices must be integers, not
     'str'` when handed a failed upstream result.

   Re-running the identical goal after these fixes: the model correctly omitted `run_id`, the
   pipeline ran for real, and the report's numbers matched the actual tool results exactly
   (cross-checked figure by figure, not just eyeballed) — `had_tool_errors` still correctly showed
   `true` because one *earlier* attempt within the same run failed (the model retried with corrected
   feature names and it worked the second time — a normal, honest instance of an error being
   surfaced and having the agent recover from it, exactly the intended behavior, as opposed to the
   first run's outright fabrication).
5. **Groq/llama-3.3-70b reliably reads a large tool *result* but sometimes fails to reproduce it
   verbatim as a tool *call* argument a couple of turns later** — observed directly: the model
   passed the literal string `"classification_results"` as `compare_models`'s `results` argument
   instead of the actual JSON object `train_classification` had just returned. Rather than assume
   this always happens (a later, cleaner run showed the model *did* echo the JSON back correctly on
   its own), added a self-healing fallback specifically in `agent/pipeline_agent.py`'s
   orchestration loop (not in the shared `agent/tools.py`, to avoid weakening the explicit-argument
   contract for the MCP server / other callers): every successful `train_*` call's real output is
   cached by task, and if the model's own `compare_models` call doesn't supply a valid dict, the
   cached value is substituted transparently before the tool actually runs.
6. **A real, previously-undiscovered gap in `utils/api_client.py` — not new agent code — was
   found because the agent's `compare_models` result showed `"mlflow_registration": null"` for a
   comparison call that included `dataset_name`.** Investigating (by reproducing the exact same
   request directly against `/comparison/classification` with `dataset_name` manually included,
   which *did* register the model) revealed that `api_client.compare_regression/classification/
   clustering` — built in section 2, used by both `pages/4_Comparison.py` and now
   `agent/tools.py` — never accepted or forwarded a `dataset_name` parameter at all. This meant
   `POST /comparison/*`'s server-side MLflow-registration-on-comparison-winner (section 4) had
   **never actually been reachable from the real Streamlit app**, only from raw curl calls made
   during section 4's own verification — a gap between two sections built in different sessions
   that neither session's testing happened to cross. Fixed in two places: `api_client.py`'s three
   `compare_*` functions now take and forward an optional `dataset_name`, and `pages/4_Comparison.py`
   was updated to actually pass `st.session_state.dataset_name` through at all three call sites —
   completing wiring that section 4 believed was already done. Also hardened `agent/tools.py`'s
   `compare_models` to surface `mlflow_error` from a failed registration attempt instead of
   silently swallowing it (`except api_client.ApiError: pass`), which is what made this gap harder
   to notice in the first place — a `mlflow_registration: null` with no accompanying error message
   looks identical to "registration wasn't attempted" and "registration was attempted and failed."

**Verification performed** (real Groq calls, real warehouse data, real MCP protocol — not written
and assumed correct):
- Retrieval: ran 4 different EDA-style queries against the real 33-chunk corpus and inspected the
  actual top-3 results for topical relevance before building the co-pilot on top of it.
- Citation verification: confirmed both directions — a real grounded answer's legitimate citations
  pass, and a hand-constructed fabricated citation against the same source chunk is correctly
  flagged — not just "it ran without erroring."
- `POST /copilot/{name}/explain` and `/recommend` — real HTTP calls against a real cleaned dataset,
  inspected full grounded answers and confirmed `unverified_citation_markers` was empty.
- `POST /agent/run` — four full end-to-end runs against real live services while iterating on the
  hallucination fix (see Mistakes #4–6 above), the last of which was cross-checked figure-by-figure
  against the actual tool-call trace, not just read for plausibility.
- MCP server: a real client session over the actual stdio transport (`mcp.client.stdio
  .stdio_client` + `ClientSession`) — `initialize()`, `list_tools()` (confirmed all 11 tools
  registered), and `call_tool()` for both `eda_summary` (real warehouse data returned) and
  `explain_eda_finding` (a full grounded, cited answer returned over the protocol) — not just that
  `agent/mcp_server.py` imports without error.
- Warehouse cleanup: dropped all `*_test`/`agent_demo*` tables and their `etl_runs` rows after
  verification, same hygiene as sections 3–4.

---

## Infra housekeeping — Git LFS

**Progress:** Git LFS set up for large binary files that aren't model artifacts.

**Decisions**
- **LFS covers what DVC doesn't**: DVC (section 4) already owns model artifacts specifically
  (`models/*.pkl`, opt-in per-file via `dvc add`). LFS's `.gitattributes` patterns
  (`*.pdf`/`*.pptx`/`*.parquet`/`*.h5`/`*.onnx`/`*.pt`/`*.ckpt`) deliberately exclude `.pkl` to
  avoid two systems claiming the same file type — LFS handles other large binaries (starting with
  the 3MB course report PDF), DVC handles trained models.
- **Set up forward-only, not migrated retroactively**: `git lfs migrate import` would rewrite every
  commit's hash from the one that first added the PDF onward (likely the repo's very first commit),
  requiring a force-push to the remote to publish — a destructive operation on shared history that
  wasn't asked for. Instead: added the LFS tracking pattern, then `git rm --cached` +
  re-`git add` the already-committed PDF, which stages it fresh through the now-active LFS filter.
  This is a normal, safe, additive commit — confirmed via `git lfs ls-files` and a diff showing the
  committed blob shrink from 3,127,231 bytes to a 132-byte pointer. The pre-LFS full-size blob
  still exists in older commits' history (mildly wasteful, but nothing was destroyed or force-pushed).

---

## Section 8 (partial) — Prometheus + Grafana

**Progress:** Metrics/dashboards half of observability added — `prometheus-fastapi-instrumentator`
on the API, a `prometheus` scrape target, and a provisioned `grafana` dashboard, all wired into
docker-compose. Langfuse/OTel (the LLM-tracing half of section 8) remain open, tracked separately.

**Decisions**
- **Split section 8 in two, same reasoning as DVC/LFS**: Prometheus/Grafana covers request-level
  metrics for the FastAPI backend; Langfuse (not yet done) is the right tool for LLM/agent traces
  specifically. One tool per concern, not one stretched over both.
- **File-based Grafana provisioning**, not clicking through the UI and exporting — datasource
  (`observability/grafana/provisioning/datasources/datasource.yml`) and dashboard
  (`observability/grafana/provisioning/dashboards/dashboard.yml` +
  `observability/grafana/dashboards/datoscope-api.json`) are both version-controlled and rebuild
  identically from a clean `docker compose up`.
- **Dashboard has 4 panels**: request rate by endpoint, p95 latency by endpoint, error rate
  (non-2xx) by endpoint, and requests by status code — all driven by the instrumentator's default
  `http_requests_total` / `http_request_duration_seconds_bucket` metrics, no custom instrumentation
  needed for this pass.

**Mistakes & Fixes**
- Planned a 5th panel, "in-progress requests", assuming `prometheus-fastapi-instrumentator` exposes
  an in-progress gauge by default (it does for some other frameworks' equivalents). Checked the
  actual installed version's `/metrics` output and `metrics` module exports via `inspect`/`dir()` —
  no such metric exists here. Fixed by swapping that panel for "requests by status code" using the
  confirmed-present `http_requests_total` status label instead of guessing from memory of the
  library's docs.
- **Found and fixed a pre-existing, unrelated bug while verifying**: the API container crash-looped
  on startup with `ModuleNotFoundError: No module named 'agent'`. Root cause: `Dockerfile` was
  never updated when the `agent/` package was added in section 7 — it copies `api/` and `etl/` but
  not `agent/`, so `api/routers/agent.py`'s `from agent.pipeline_agent import run` failed at import
  time in the container even though it worked fine outside Docker (repo root was on `sys.path`
  locally). Fixed with one line, `COPY agent/ ./agent/`, added right after the existing `COPY api/`
  line. This had nothing to do with Prometheus/Grafana directly, but the API container needed to
  actually be up for Prometheus to have anything to scrape, so it surfaced here.

**Verification**
- Rebuilt and brought up `minio`/`warehouse`/`mlflow`/`api`/`prometheus`/`grafana` via
  `docker compose up -d --build`; confirmed all containers reach `Up`/`healthy`, not just `Created`.
- API: `curl localhost:8000/metrics` returns real `http_requests_total` counter lines.
- Prometheus: `GET /api/v1/targets` shows `job="datoscope-api"` with `health: "up"`;
  `GET /api/v1/query?query=up{job="datoscope-api"}` returns a live `1` sample;
  `http_requests_total{job="datoscope-api"}` returns real series after generating traffic — not
  just that the container is running, but that it's actually pulling real data from the API.
- Grafana: `GET /api/health` reports `database: ok`; `GET /api/datasources` (authenticated) shows
  the Prometheus datasource provisioned and marked default; `GET /api/search` shows the
  `DatoScope API` dashboard loaded under the `DatoScope` folder — confirming the provisioning files
  were actually read and applied, not just present on disk unused.

---

## Section 5 — AWS / Terraform

**Progress:** Terraform modules for S3, RDS, ECR, and IAM, wired together in a root config, plus a
separate LocalStack-backed environment used to `apply` (not just `validate`) the S3/IAM modules
for real.

**Decisions**
- **Four modules, one root config**: `terraform/modules/{s3,rds,ecr,iam}` each own one AWS service,
  `terraform/main.tf` instantiates and wires them (e.g. IAM's policies reference S3's bucket ARNs
  via `module.s3.bucket_arns`). Standard Terraform module boundary — each one is independently
  testable/replaceable.
- **S3 bucket names get a random suffix** (`random_id.suffix`, 4 bytes): S3 bucket names are
  globally unique across *all* AWS accounts, not just this one, so `datoscope-dev-raw` would
  collide with anyone else who's ever created that name. The random suffix avoids needing the user
  to hand-pick a unique prefix.
- **RDS defaults to the account's default VPC** (`data.aws_vpc.default` / `data.aws_subnets.default`
  when `vpc_id`/`subnet_ids` aren't passed): this is a free-tier dev setup, not a production network
  topology with custom VPC/subnet design — todo.md section 5 scopes this as "keep the live AWS
  footprint minimal", so a bespoke VPC module would be scope creep for what this needs to prove.
- **IAM roles are EKS-IRSA-ready but not yet IRSA-bound**: section 6 (Kubernetes/EKS) doesn't exist
  yet, so there's no OIDC provider to trust. `eks_oidc_provider_arn`/`_url` default to `null`, and
  the roles fall back to an account-root `sts:AssumeRole` trust policy as a placeholder (IAM roles
  can't exist with no principal at all). Once section 6 provisions a real EKS cluster, its OIDC
  provider ARN/URL get passed in and the roles become properly IRSA-scoped to specific K8s service
  accounts — no restructuring needed, just filling in two variables that were designed for this.
  Recorded as `irsa_enabled` output so it's visible which mode is active.
- **ECR gets a lifecycle policy from the start** (expire untagged after 1 day, keep last 10 tagged):
  the CD pipeline (todo.md section 4's deferred CD job, being unblocked now) will push an image on
  every merge to main, and unbounded image retention is a real cost/clutter problem, not a
  hypothetical one — cheap to set up now vs. bolting on later.
- **Chose LocalStack over "validate only" for a stronger check**: `terraform validate` only checks
  HCL syntax and type-correctness — it doesn't catch e.g. a broken cross-resource reference that's
  syntactically valid but wrong, or a provider-rejected argument combination. LocalStack runs a real
  `apply`/`destroy` cycle with real state and real (emulated) AWS API calls, without needing actual
  AWS credentials or spending money — a genuinely stronger verification for a session built around
  "verify against real running services," not just a nice-to-have.
- **Separate root module for LocalStack** (`terraform/envs/localstack/`), not a flag on the main
  provider config: keeps the LocalStack endpoint overrides completely isolated from the config that
  will eventually point at real AWS — no risk of a stray env var accidentally routing a real `apply`
  at LocalStack or vice versa. It re-uses the same `modules/s3` and `modules/iam` source, so what
  gets tested is the actual module code, not a parallel copy of it.

**Mistakes & Fixes**
- `brew install terraform helm kind git-lfs` failed with "No available formula with the name
  'terraform'" — HashiCorp pulled Terraform from homebrew-core after their license change (BSL) and
  moved it to their own tap. Fixed with `brew tap hashicorp/tap` then
  `brew install hashicorp/tap/terraform helm kind git-lfs`.
- First LocalStack attempt used the `latest` tag and failed immediately: "License activation
  failed! ... No credentials were found" — `latest` now resolves to an image that expects a Pro auth
  token even for community-only usage. Fixed by pinning to `localstack/localstack:3.8.1`, a known
  community-edition release, which started clean with no token needed.
- Planned to `apply` all four modules (s3/rds/ecr/iam) against LocalStack, but `ecr` failed with
  `API for service 'ecr' not yet implemented or pro feature` — checked LocalStack's own coverage
  docs reference in the error message rather than assuming; ECR emulation is Pro-only in LocalStack
  community edition. Dropped `ecr` from the LocalStack env (and `rds`, which has known-flaky
  community support) rather than working around it, and documented why in
  `terraform/envs/localstack/README.md` so it's not mistaken for an oversight later.
- `.gitignore`'s original Terraform patterns (`terraform/.terraform/`, `terraform/*.tfstate`, etc.)
  only matched the root `terraform/` directory, not the nested `terraform/envs/localstack/` one —
  caught by `git status` showing `.terraform/`/`terraform.tfstate` under the nested path as
  untracked before staging. Fixed by switching to `**/`-prefixed globs
  (`terraform/**/*.tfstate`, `**/.terraform/`, etc.) that match at any depth.

**Verification**
- `terraform fmt -recursive` + `terraform validate` pass clean on the main root config (all four
  modules).
- Real `terraform apply -auto-approve` against a live LocalStack container (`s3` + `iam` modules,
  21 resources) succeeded; verified the actual resources exist independently via the AWS CLI
  (`aws --endpoint-url=http://localhost:4566 s3 ls` shows all 4 buckets, `iam list-roles` shows
  both roles, `iam list-role-policies` shows the attached S3 policy, `s3api get-bucket-versioning`
  confirms versioning is actually `Enabled` — not just that Terraform's state file says so) —
  then `terraform destroy -auto-approve` cleanly removed all 21 resources, and the LocalStack
  container was stopped/removed afterward (no lingering test infra).
- No real AWS `apply` attempted — no AWS credentials exist in this environment or `.env`, and doing
  so would need explicit user authorization for real cloud spend, which wasn't given.

---

## Section 6 — Kubernetes

**Progress:** Raw K8s manifests for the API/Streamlit app, plus Airflow deployed on Kubernetes via
the official Helm chart with `KubernetesExecutor`, both verified on a real local `kind` cluster.

**Decisions**
- **Raw manifests for the app, not a Helm chart**: two straightforward Deployments/Services plus
  one Ingress don't need templating — a chart would add a layer of indirection with no real payoff
  here. `kustomization.yaml` ties them together for a single `kubectl apply -k`. Airflow, by
  contrast, uses the official chart because writing a from-scratch scheduler/webserver/RBAC/DB-init
  setup would be reinventing a well-maintained wheel for no benefit.
- **`secret.example.yaml` + gitignored `secret.yaml`**, same pattern as `terraform.tfvars.example`
  from section 5 — consistent placeholder-file convention across the repo for anything that needs
  real credentials to run.
- **DAGs baked into the Airflow image**, not git-sync or a PVC: `Dockerfile.airflow` now `COPY`s
  `airflow/dags/` in addition to `etl/`/`utils/` (previously only available via docker-compose's
  bind mount). This is the standard "immutable image" deployment pattern for the official chart
  when you don't want a git-sync sidecar, and it doesn't change docker-compose's local-dev
  behavior at all — the bind mount there still overlays the baked-in copy at container start, so
  editing a DAG locally still doesn't need a rebuild.
- **Airflow's metadata DB is a separate concern from the app's data warehouse** (RDS in section 5):
  same split as the local docker-compose stack (`airflow-metadata` vs. `warehouse`, two different
  Postgres instances/databases). Not conflated here either.
- **Chart pinned to version `1.15.0`**, not the repo default: the unversioned chart now defaults to
  targeting Airflow 3.x (renamed `webserver`→`api-server`, different config surface), which doesn't
  match this project's `apache/airflow:2.9.3` image from section 4. `1.15.0` is the release whose
  `appVersion` is exactly `2.9.3` (checked via `helm search repo apache-airflow/airflow --versions`
  rather than guessing).
- **`values-kind.yaml` vs `values-eks.yaml`**, mirroring `terraform/envs/localstack/` vs. the main
  Terraform config: same chart, environment-specific overrides isolated into separate files rather
  than one file trying to serve both with conditionals — image source, metadata DB, and IRSA
  annotations differ between "disposable local smoke test" and "real cluster."
- **Disabled Celery/Redis/Flower/statsd/triggerer** in both values files: `KubernetesExecutor`
  launches one pod per task and has no use for a Celery worker fleet or its broker; statsd/flower
  are optional and not needed to prove the executor works; triggerer is for deferrable operators,
  which this DAG doesn't use. Keeps the local smoke-test footprint small and the values files
  honest about what's actually being exercised.

**Mistakes & Fixes**
- First Airflow install used the Helm repo's default (unversioned/`latest`-resolving) chart, which
  targeted Airflow 3.x — pods came up as `api-server`/`dag-processor` instead of the expected
  `webserver`/`scheduler`, a structural mismatch with the 2.9.3 image, not something that could be
  patched with a values tweak. Diagnosed by comparing the running pod names against what the chart
  version should produce, then fixed by uninstalling and reinstalling pinned to `--version 1.15.0`.
- The chart's bundled Postgres subchart failed with `ImagePullBackOff` on
  `bitnami/postgresql:16.1.0-debian-11-r15` — Bitnami has since pulled that exact tag from Docker
  Hub (confirmed via `kubectl describe pod`'s event log: `not found`, not a network/auth issue).
  Fixed by dropping the bundled subchart entirely (`postgresql.enabled: false`) and standing up a
  plain `postgres:16` Deployment/Service instead (`k8s/airflow/metadata-postgres.yaml`), pointed to
  via the chart's `data.metadataConnection` values — same image this project already uses
  everywhere else for Postgres, so no new moving part, just not the chart's default.
- After fixing the DB, the DAG failed to load at all: `ModuleNotFoundError: No module named 'etl'`
  from `airflow dags list-import-errors`. Root cause: docker-compose's Airflow services set
  `PYTHONPATH=/opt/airflow` so `etl/` (copied to `/opt/airflow/etl`) is importable from a DAG file
  under `/opt/airflow/dags/` — this env var was never carried over into the Helm values. Fixed by
  adding it to both `values-kind.yaml` and `values-eks.yaml`'s `env:` list; re-ran
  `airflow dags list` afterward and confirmed zero import errors before triggering anything.
- (Not a mistake, but worth recording): task pods launched by the KubernetesExecutor are deleted
  quickly after finishing, so `kubectl logs <pod>` raced the pod's own lifecycle and returned
  `NotFound` — the pods had already been garbage-collected by the time I went to fetch their
  stdout. Confirmed the launch happened anyway via `kubectl get events` (Scheduled → Pulled →
  Created → Started, plus the task's own `up_for_retry` state with real start/end timestamps ~10s
  apart) rather than needing the raw log output — good enough evidence that pods really launched,
  ran, and failed at the connection step, not that they never ran at all.

**Verification**
- App: `kind create cluster` (with the ingress-ready node label + port mappings ingress-nginx's own
  kind docs specify) + a real ingress-nginx controller install, waited for it to report `ready`.
  Built and `kind load docker-image`'d the actual app image (already verified working in section
  8's Prometheus/Grafana work, agent/ COPY fix included). `kubectl apply -k k8s/base/`: both
  Deployments reached `available`; `curl http://localhost:8090/api/health` (through the real
  Ingress, not a port-forward shortcut) returned `{"status":"ok"}`; `curl
  http://localhost:8090/_stcore/health` returned `ok` for Streamlit.
- Airflow: scheduler + webserver reached `Running`/`2/2` and `Running`/`1/1` respectively;
  `airflow dags list` showed `datoscope_etl_dag` with no import errors; `airflow dags trigger`
  produced a real queued run; `kubectl get events` confirmed the KubernetesExecutor created,
  scheduled, and started real pods per task (not simulated) — the specific mechanism this todo item
  claims. Task-level success was explicitly out of scope (no MinIO/warehouse in this cluster) and
  documented as such rather than silently skipped.
- Cleaned up afterward: DAG re-paused, `helm uninstall`, namespaces deleted, `kind delete cluster`
  — no lingering local test infra, same hygiene as the LocalStack teardown in section 5.
- EKS path: documented and reviewed against the real `terraform output` shapes from section 5, but
  not applied — no EKS cluster exists and provisioning one has a real hourly cost that wasn't
  authorized.

---

## Section 4 wrap-up — CD

**Progress:** The CD job deferred at the end of section 4 (see that section's entry above) is now
built — a second job in `.github/workflows/ci.yml`, now that sections 5/6 give it a real ECR
registry and real K8s manifests/Helm chart to target.

**Decisions**
- **One workflow, two jobs**, not a separate `cd.yml`: matches what todo.md scoped from the start,
  and keeps the CD job's `needs: lint-and-test` gate simple (same-workflow job dependencies, no
  cross-workflow triggering).
- **OIDC role assumption (`aws-actions/configure-aws-credentials` + `role-to-assume`), not static
  AWS access keys as a secret**: long-lived AWS credentials sitting in GitHub Secrets are a
  standing risk (leak once, valid forever until manually rotated); an OIDC trust policy scoped to
  this repo means there's no long-lived secret to leak in the first place. This is also
  AWS/GitHub's own current recommended pattern, not a nonstandard choice.
- **`kubectl set image` for the app, `helm upgrade` for Airflow** — matches how each was actually
  deployed in section 6 (raw manifests vs. chart), rather than switching both to Helm or both to
  raw `kubectl apply` for uniformity's sake; each deploy step mirrors the tool that owns that
  resource.
- **Every name the job references is the real one**, cross-checked against sections 5/6 rather than
  written from memory: ECR repo naming (`datoscope-api`, `datoscope-airflow`) matches
  `terraform/modules/ecr`'s `repository_names` default; namespace/Deployment/container names
  (`datoscope`/`datoscope-api`/`api`, `datoscope-streamlit`/`streamlit`) match `k8s/base/*.yaml`
  exactly (verified against section 6's actual kind deployment, not just read off the YAML); the
  Airflow Helm release name/chart version (`datoscope-airflow`, `1.15.0`) match what was verified
  working in section 6, including the chart-version pin that section 6 discovered was necessary.

**Verification**
- Not run end-to-end — no EKS cluster or AWS credentials exist in this environment (same
  constraint noted throughout sections 5/6), and creating either would need explicit authorization
  for real cloud cost/access that wasn't given.
- What *was* done: every AWS/K8s resource name/path the job references was cross-checked against
  the real Terraform module outputs (section 5) and the real k8s manifests as actually deployed and
  verified in section 6 (not just read from the YAML source) — the closest thing to verification
  possible without a live target, and a meaningfully stronger check than writing the job from the
  chart/module docs alone.
