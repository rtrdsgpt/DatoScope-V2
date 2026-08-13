# DatoScope V2

DatoScope is a multipage Streamlit application for synthetic data generation, dataset upload, preprocessing, exploratory data analysis, supervised learning, clustering, and model comparison.

This is a solo continuation of a group course project originally built for the MA5755 course
(IIT Madras) by Tanmoy (MA25M026), Mehak (MA25M016), and Aritra (MA25M005), under the supervision
of Prof. Rakhi Singh. V2 extends the original classical-ML/EDA app with an ETL pipeline, API layer,
MLOps, cloud/Kubernetes deployment, and an AI co-pilot layer — see `todo.md` for the roadmap.

#### (Now supporting Light and Dark Modes!)

## Highlights

- Generate datasets inside the app for regression, classification, or clustering
- Upload a single dataset or separate train/test files
- Optionally create a train/test split during dataset generation
- Clean train and test data with missing-value handling, outlier removal, duplicate removal, scaling, and categorical encoding
- Run EDA with summary statistics, distributions, Q-Q plots, correlations, scatter plots, and explainable variance ranking
- Train regression models: Linear Regression, Ridge, Lasso
- Train classification models: Logistic Regression, Random Forest, KNN
- Visualize individual trees from trained Random Forest classifiers
- Run clustering models: K-Means, DBSCAN, Hierarchical Clustering
- Compare regression, classification, and clustering results in dedicated pages
- Download trained supervised models as `.pkl`

## Data Input Modes

The sidebar supports three workflows:

1. `Generate Dataset`
   Create synthetic regression, classification, or clustering datasets with controls for:
   - dataset type
   - sample count
   - noise
   - number of clusters / arms
   - number of features
   - target column name
   - optional generated train/test split
   - random seed

2. `Upload Single File`
   Upload one file and let the app create an internal train/test split during supervised modeling.
   Uploaded datasets can also use categorical encoding during preprocessing.

3. `Upload Train/Test`
   Upload a train file and an optional test file.
   If the test file is present, the app uses it directly instead of creating a split.

## Pages

- `app.py`
  Streamlit entry point; builds the sidebar navigation for the pages below
- `pages/1_EDA.py`
  Exploratory data analysis
- `pages/2_Supervised_Modeling.py`
  Regression and classification workflows, random forest controls, and tree visualization
- `pages/3_Clustering.py`
  Clustering workflows, cluster-size plots, and optional ground-truth metrics
- `pages/4_Comparison.py`
  Model comparison dashboard with improved regression scoring and explicit winner-selection logic

## Project Structure

```text
DatoScope/
├── app.py
├── data_loader.py
├── pages/
│   ├── 1_EDA.py
│   ├── 2_Supervised_Modeling.py
│   ├── 3_Clustering.py
│   └── 4_Comparison.py
├── utils/
│   ├── api_client.py
│   ├── app_state.py
│   ├── comparison.py
│   ├── data_input.py
│   ├── generators.py
│   ├── io.py
│   ├── modeling.py
│   ├── preprocessing.py
│   └── ui.py
├── tests/
├── scripts/
│   ├── 01_generate_data.py
│   ├── 02_clean_data.py
│   ├── 03_eda.py
│   └── 04_visualization.py
├── etl/
│   ├── config.py
│   ├── storage.py
│   ├── extract.py
│   ├── transform.py
│   ├── validate.py
│   ├── load.py
│   └── pipeline.py
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── serialization.py
│   ├── tracking.py
│   └── routers/
│       ├── datasets.py
│       ├── eda.py
│       ├── modeling.py
│       ├── clustering.py
│       ├── comparison.py
│       └── models.py
├── airflow/
│   └── dags/
│       └── datoscope_etl_dag.py
├── models/                    # DVC-tracked Production model exports (empty until you export one)
├── .github/workflows/ci.yml
├── docker-compose.yml
├── Dockerfile               # api + streamlit (shared)
├── Dockerfile.airflow
├── Dockerfile.mlflow
├── train.py
├── requirements.txt
├── requirements-dev.txt
└── requirements-etl.txt
```

## Setup

Use Python 3.11 for the most reliable dependency compatibility.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app.py
```

## Run The Offline Pipeline

```bash
python train.py
```

This pipeline runs:

1. synthetic dataset generation
2. cleaning and preprocessing
3. EDA reporting
4. static plot generation

## ETL Pipeline

`etl/` is the explicit extract → transform → validate → load pipeline: it lands raw data
(generated in-app, user-uploaded, or pulled from Kaggle) in an S3-compatible "raw" zone,
cleans it with the same logic `utils/preprocessing.py` uses, validates it with Great
Expectations (schema/null/range checks — the pipeline fails loudly on violation rather than
loading bad data), and loads the result into a Postgres warehouse table.

```bash
cp .env.example .env   # then fill in KAGGLE_API_TOKEN (see comments in the file)

docker compose up -d minio warehouse   # MinIO (raw/processed zones) + Postgres (warehouse)

pip install -r requirements.txt
python -c "from etl.pipeline import run_pipeline; \
  print(run_pipeline('kaggle', dataset_name='iris', extract_kwargs={'handle': 'uciml/iris'}))"
```

`run_pipeline` also accepts `source='generated'` (pass `extract_kwargs={'df': ..., 'generator_meta': ...}`)
and `source='uploaded'` (pass `extract_kwargs={'filename': ..., 'raw_bytes': ...}`).

### Orchestrating with Airflow

The same pipeline functions run as an Airflow DAG (`airflow/dags/datoscope_etl_dag.py`),
scheduled daily against the Kaggle source:

```bash
docker compose up -d                 # brings up MinIO, warehouse, and Airflow (LocalExecutor)
# Airflow UI at http://localhost:8080 (admin/admin)

# or run/validate the DAG directly without the webserver:
docker compose run --rm airflow-init -c "airflow dags test datoscope_etl_dag $(date +%F)"
```

## API

`api/` is a FastAPI backend exposing dataset ingestion, EDA, preprocessing, modeling, and
clustering as REST endpoints, reading from the Postgres warehouse (via `etl/load.py`) instead of
local files. **Streamlit is a client of this API** (`utils/api_client.py`) — the sidebar's
Generate/Upload/Clean & Preprocess and all four pages call it over HTTP rather than calling
`utils.generators`/`utils.preprocessing`/`utils.modeling` in-process. Run both together:

```bash
docker compose up -d minio warehouse
uvicorn api.main:app --reload &          # docs at http://localhost:8000/docs
streamlit run app.py
```

Streamlit finds the API at `API_BASE_URL` (env var, defaults to `http://localhost:8000`).

Typical flow: `POST /datasets/generate` (or `/upload`, `/kaggle`) lands raw data and returns a
`run_id`, then `POST /datasets/{name}/clean` transforms + validates + loads it into the warehouse
(same as the ETL pipeline's stages) and returns a *new* `run_id` — a raw extract can be cleaned
more than once (different parameters), each producing its own queryable warehouse run rather than
colliding. From there:

- `GET /datasets/{name}/raw|data` — row-level data (raw zone / warehouse respectively)
- `GET /eda/{name}/summary|missing|distributions|boxplot|qq|correlation|variance` — EDA stats
- `POST /modeling/{name}/regression|classification` — train/evaluate (optionally against a
  separate `test_dataset_name`/`test_run_id`), returns metrics + a `model_id`;
  `GET /modeling/download/{model_id}` returns the fitted model as a `.pkl`
- `POST /clustering/{name}` — train/evaluate KMeans/DBSCAN/Hierarchical + a 2D projection for
  plotting + the scaled feature matrix (used client-side for the dendrogram/elbow diagnostics)
- `POST /comparison/regression|classification|clustering` — winner-selection over a set of
  already-computed model metrics (shared with the API via `utils/comparison.py`); also registers
  the winner into the MLflow Model Registry (see MLOps below)
- `GET /models/{name}/versions`, `POST /models/{name}/promote` — MLflow Model Registry (Staging →
  Production)

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                    # unit tests only — pure functions, no external services, ~3s
pytest -m integration     # also exercises the real ETL pipeline end to end
```

The integration tests (`tests/test_etl_load.py`, `tests/test_etl_pipeline.py`) need a live
warehouse — `docker compose up -d minio warehouse` first, or they skip automatically. Everything
else (`utils/preprocessing.py`, `utils/modeling.py`, `utils/generators.py`, `utils/comparison.py`,
and the ETL `transform`/`extract`/`validate` stages) runs with no external services: S3 calls are
mocked with `moto`, and Great Expectations runs in an ephemeral in-process context.

`tests/test_etl_validate.py` is the data-quality test layer specifically — it feeds known-bad
fixtures (missing required columns, excess nulls, out-of-range values, empty tables) through
`etl/validate.py` and asserts they're rejected, not just that clean data passes.

## MLOps

### Full dev stack

`docker compose up -d` now brings up everything — MinIO, the warehouse, MLflow, Airflow, the
FastAPI backend, and Streamlit — as one command:

```bash
docker compose up -d --build
# Streamlit:  http://localhost:8501
# API docs:   http://localhost:8000/docs
# MLflow UI:  http://localhost:5001
# Airflow UI: http://localhost:8080
# MinIO console: http://localhost:9001
```

`api` and `streamlit` share one `Dockerfile` (different `command:` per service in
`docker-compose.yml`). Running the API/Streamlit directly on the host (as in the sections above)
still works too — nothing requires the containers.

### MLflow — experiment tracking + Model Registry

Every model trained via `POST /modeling/{name}/regression|classification` or
`POST /clustering/{name}` is logged as its own MLflow run (params, metrics, and the model
artifact) — "log every model-comparison run" from `todo.md` section 4. The backend store is a
`mlflow` database in the same warehouse Postgres; the artifact store is a `datoscope-mlflow`
bucket in the same MinIO — no separate infrastructure, both auto-created on first start by
`docker/mlflow_entrypoint.py`.

When `POST /comparison/regression|classification|clustering` is called with a `dataset_name`, the
winning model (matched via the `mlflow_run_id` the modeling/clustering endpoint attached to it) is
registered into the Model Registry as `<dataset_name>__<task>` and moved to **Staging**
automatically — the comparison that names a winner *is* the promotion trigger. Moving a model from
Staging to **Production** is a separate, deliberate call:

```bash
curl -X POST http://localhost:8000/models/<dataset_name>__<task>/promote
curl http://localhost:8000/models/<dataset_name>__<task>/versions
```

### DVC — Production model artifacts

DVC tracks model *artifacts* specifically (not the processed datasets, which live in the
warehouse, not flat files — see `etl/load.py`), with MinIO as the S3 remote
(`datoscope-models` bucket, configured in `.dvc/config`). Exporting a model is a deliberate,
reviewable step, not something the API does automatically:

```bash
python scripts/05_export_production_model.py <dataset_name>__<task>   # pulls the Production version from MLflow
dvc add models/<dataset_name>__<task>.pkl
git add models/<dataset_name>__<task>.pkl.dvc models/.gitignore
git commit -m "Track <dataset_name>__<task> Production model"
dvc push
```

`dvc pull` restores tracked model files from MinIO on a fresh checkout.

### CI

`.github/workflows/ci.yml` runs on every push/PR to `main`: `ruff check .` (lint), then `pytest`.
No services are started in CI, so the integration tests self-skip (see Testing above) — CI covers
the full pure-function suite; run `pytest -m integration` locally against
`docker compose up -d minio warehouse` for the rest. CD (build → push to ECR → deploy to
Kubernetes) is deferred to sections 5–6, which provision the registry and cluster it would target.

## Supported File Types

| Extension | Notes |
|---|---|
| `.csv` | Standard CSV upload |
| `.xlsx` / `.xls` | Excel upload with simple header detection |
| `.zip` | Must contain one CSV |
| `.data` | Headerless comma- or whitespace-delimited files |

## Notes

- Generated datasets can now be created specifically for regression, classification, or clustering.
- The sidebar includes a selected ML task control so the interface can stay focused on one task at a time.
- Uploaded datasets can encode categorical variables using One-Hot or Label encoding during preprocessing.
- Classification is inferred from the selected target column but can also be chosen manually in the supervised modeling page.
- Classification train/test splitting now falls back safely when a class has too few samples for strict stratification.
- Model export currently supports supervised models.
- Clustering runs on the train dataset only when a separate test file is present.
- If a dataset contains ground-truth `label` values, clustering comparison can also report Fowlkes-Mallows and Rand Index scores.
- Regression comparison now considers both predictive quality and generalization instead of choosing winners from raw test R² alone.
- Clustering comparison now selects the best algorithm by counting how many of the five tracked clustering metrics each model wins.
