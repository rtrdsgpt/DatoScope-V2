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
│   ├── 04_visualization.py
│   ├── 05_export_production_model.py
│   └── 06_build_rag_corpus.py
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
│       ├── models.py
│       ├── copilot.py
│       └── agent.py
├── agent/
│   ├── groq_client.py       # multi-key rotation for Groq's free tier
│   ├── rag.py                # retrieval over rag_corpus.pkl (built by scripts/06_...)
│   ├── copilot.py             # grounded EDA explanations + preprocessing recommendations
│   ├── tools.py                 # shared tool functions — used by both of the below
│   ├── schema_gen.py             # generates tool-calling JSON schemas from tools.py's signatures
│   ├── pipeline_agent.py          # autonomous goal -> pipeline -> report agent
│   └── mcp_server.py                # exposes agent/tools.py + copilot for Claude Desktop
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

This repo uses [Git LFS](https://git-lfs.com) for large binary files (the course report PDF, and
any `.parquet`/`.pt`/`.h5`/etc. that show up later — see `.gitattributes`; model artifacts go
through DVC instead, not LFS — see MLOps below). Install it once per machine, then clone/pull
normally:

```bash
brew install git-lfs   # or see git-lfs.com for other platforms
git lfs install
git clone https://github.com/rtrdsgpt/DatoScope-V2.git
```

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
- `POST /copilot/{name}/explain|recommend`, `POST /agent/run` — see AI Co-Pilot & Agent below

## AI Co-Pilot & Agent

`agent/` is the differentiator layer (`todo.md` section 7) — a grounded LLM co-pilot, an
autonomous pipeline agent, and an MCP server, all built on the same tool implementations
(`agent/tools.py`, thin wrappers around `utils/api_client.py`) and running on Groq (free tier;
`GROQ_API_KEY` in `.env`, comma-separate multiple keys for round-robin rotation when one hits a
rate limit — see `agent/groq_client.py`).

### Grounded co-pilot

`POST /copilot/{name}/explain` and `POST /copilot/{name}/recommend` answer questions about a
cleaned dataset's EDA findings and recommend preprocessing steps, grounded in a RAG corpus built
from the *installed* sklearn/scipy docstrings (`scripts/06_build_rag_corpus.py` — always in sync
with the pinned versions, no scraping). Citations are inline `[S1]`-style markers the model is
required to use; each cited quote is then deterministically checked against the actual retrieved
source text (`agent/copilot.py`'s `_verify`) — a real hallucination guard, not just a prompt
instruction the model may or may not follow. `unverified_citation_markers` in the response flags
anything that didn't check out.

```bash
python scripts/06_build_rag_corpus.py   # once, or whenever CORPUS_SYMBOLS/sklearn version changes
curl -X POST http://localhost:8000/copilot/<dataset_name>/explain \
  -H "Content-Type: application/json" -d '{"question": "Why is x1 skewed?"}'
```

### Autonomous pipeline agent

`POST /agent/run` takes a natural-language goal (e.g. *"find the best classifier for this churn
dataset"*) and runs a Groq tool-calling loop over `agent/tools.py` — generate/pull a dataset,
clean it, EDA, train, compare — producing a written report. The response's `had_tool_errors` flag
is a deterministic integrity check independent of the model's own narrative: an LLM asked to
summarize a run where a step failed will sometimes write a confident success story around
fabricated numbers instead of reporting the failure (this was caught and fixed during
verification — see `log.md`), so this flag is set by scanning the actual tool-call trace for
error results, not by trusting the report text.

```bash
curl -X POST http://localhost:8000/agent/run -H "Content-Type: application/json" -d '{
  "goal": "Generate a synthetic classification dataset and find the best classifier for it."
}'
```

### MCP server

`agent/mcp_server.py` exposes every tool (dataset ops, training, comparison, the grounded
co-pilot, and the autonomous agent itself) over stdio so Claude Desktop can drive DatoScope
directly — the pipeline's tools, not a separate reimplementation of them. Requires the API running
(`API_BASE_URL`, defaults to `http://localhost:8000`).

```json
{
  "mcpServers": {
    "datoscope": {
      "command": "/absolute/path/to/DatoScope V2/.venv/bin/python",
      "args": ["/absolute/path/to/DatoScope V2/agent/mcp_server.py"]
    }
  }
}
```

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
FastAPI backend, Streamlit, and Prometheus/Grafana — as one command:

```bash
docker compose up -d --build
# Streamlit:      http://localhost:8501
# API docs:       http://localhost:8000/docs
# MLflow UI:      http://localhost:5001
# Airflow UI:     http://localhost:8080
# MinIO console:  http://localhost:9001
# Prometheus:     http://localhost:9090
# Grafana:        http://localhost:3000 (admin/admin)
```

`api` and `streamlit` share one `Dockerfile` (different `command:` per service in
`docker-compose.yml`). Running the API/Streamlit directly on the host (as in the sections above)
still works too — nothing requires the containers.

### Prometheus + Grafana — metrics

The FastAPI backend is instrumented with `prometheus-fastapi-instrumentator`, exposing request
rate/latency/status metrics at `GET /metrics`. The `prometheus` service scrapes it every 15s
(`observability/prometheus/prometheus.yml`); the `grafana` service auto-provisions the Prometheus
datasource and a **DatoScope API** dashboard (`observability/grafana/`) with request rate, p95
latency, error rate, and status-code breakdown panels — no manual dashboard setup needed after
`docker compose up`. This covers the metrics half of `todo.md` section 8; Langfuse (LLM/agent
traces) is the other half, tracked separately.

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
`docker compose up -d minio warehouse` for the rest.

## AWS / Terraform

Infrastructure as code for the AWS deployment target — `terraform/`, four modules
(`modules/{s3,rds,ecr,iam}`) wired together in the root config:

- **S3** — `raw`/`processed`/`models`/`mlflow` buckets (versioned, encrypted, public-access
  blocked), naming mirrors the local MinIO buckets
- **RDS** — a single Postgres warehouse instance, `db.t4g.micro`/20GB (free-tier-eligible)
- **ECR** — one repo per image (`api`/`streamlit`/`airflow`), with a lifecycle policy that expires
  untagged images after 1 day and keeps the last 10 tagged
- **IAM** — separate least-privilege ETL and API roles, scoped to only the S3 buckets/actions each
  needs. EKS-IRSA-ready: pass `eks_oidc_provider_arn`/`eks_oidc_provider_url` once section 6's
  cluster exists and the roles become assumable by specific K8s service accounts; until then they
  fall back to an account-root trust placeholder (a role can't exist with no principal at all)

```bash
cd terraform
terraform init
terraform validate
# real apply needs AWS credentials + terraform.tfvars (see terraform.tfvars.example) — not run
# here, no AWS credentials exist in this environment
```

No real AWS credentials exist in this project's environment, so nothing here has been `apply`'d
against real AWS. Instead, `terraform/envs/localstack/` is a separate root module pointed at a
local [LocalStack](https://localstack.cloud) container that lets `terraform apply`/`destroy` run
for real — see its README for the walkthrough. Verified: 21 resources created (S3 buckets + IAM
roles/policies), independently confirmed via the AWS CLI against LocalStack, then destroyed
cleanly. (`ecr` is a LocalStack Pro-only feature and `rds` is flaky there, so those two are
`validate`-only.)

## Kubernetes

`k8s/` — Deployment/Service/Ingress manifests for the API + Streamlit app (`k8s/base/`), and
Airflow itself deployed via the official Helm chart with the `KubernetesExecutor` (`k8s/airflow/`,
reusing the section 1 DAG as-is, now baked into the Airflow image). Full walkthrough — both the
local `kind` path and the real EKS path — is in [`k8s/README.md`](k8s/README.md).

Verified end-to-end on a real local `kind` cluster: the app is reachable through a real
ingress-nginx controller (`curl` through the mapped port, not just `kubectl get pods`), and
triggering the Airflow DAG makes the `KubernetesExecutor` dynamically launch real per-task pods
(confirmed via `kubectl get events`) — the specific mechanism this section is about. The DAG's
tasks themselves aren't expected to succeed in this smoke test (no MinIO/warehouse deployed in the
kind cluster — that's what sections 4/5 already cover); see `k8s/README.md` for the exact
boundary. The EKS path is documented and reviewed against the real Terraform outputs from the AWS
section above, but not applied against a real cluster — none exists, and provisioning one has a
real hourly cost that wasn't authorized. The CD job below deploys to it automatically once one
does.

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
