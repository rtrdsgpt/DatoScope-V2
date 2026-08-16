# DatoScope V2 — TODO

Fork of https://github.com/tanmoyghosh704-lang/DatoScope (currently a pure classical-ML/EDA
Streamlit app — synthetic data generation, preprocessing, EDA, sklearn models, clustering, model
comparison — with zero of the RAG/Agentic/API/MLOps/testing/MCP checklist). This folder is
currently empty — first step is the fork.

**Direction (revised): "Data Platform with an AI co-pilot,"** not just "AutoML app with MLOps
bolted on." This is the project carrying ETL/AWS/Kubernetes/Langfuse — the one deliberately built
to also read as a Data Engineering project, not just Data Science, for the Data Science CV. See
`Project Plan.md` (Projects root) section 5 for the original AutoML-focused plan; this file
supersedes its scope with the additions below.

## 0. Fork setup
- [x] Fork `tanmoyghosh704-lang/DatoScope` (default branch `mehak-work`) into your own repo,
      clone it into this folder — done as a standalone repo with fresh history (not a GitHub
      fork), pushed to https://github.com/rtrdsgpt/DatoScope-V2
- [x] Clean up on fork: remove committed `.DS_Store` files, fix stale README "Project Structure"
      section (references deleted `Preprocessing.py`), fix `.devcontainer/devcontainer.json`
      (still references deleted `Preprocessing.py` in `openFiles`/`postAttachCommand`)
- [x] Add a license (currently `license: null`) — MIT

## 1. ETL pipeline (new — this is the data-engineering core)
- [x] **Extract**: ingestion stage that lands raw data in an S3 "raw" zone — cover both paths
      DatoScope already supports (user-uploaded files, synthetically generated datasets), plus
      add one real external source (e.g. a public dataset API/Kaggle) so the extract step isn't
      purely synthetic — `etl/extract.py` (`extract_generated`, `extract_uploaded`,
      `extract_kaggle` via kagglehub), raw zone is MinIO locally (S3-compatible, `etl/storage.py`)
- [x] **Transform**: refactor the existing `utils/preprocessing.py` logic into an explicit
      transform stage (cleaning, outlier handling, encoding, feature engineering) that reads from
      the raw zone and writes to a "processed" zone — not ad hoc in-app calls — `etl/transform.py`;
      also split `parse_uploaded_bytes` out of `utils/preprocessing.py` as a Streamlit-free pure
      function so the ETL package doesn't need Streamlit installed
- [x] **Data quality validation** in the transform stage — schema/null/range checks via Great
      Expectations or Pandera; fail the pipeline loudly on violation rather than silently passing
      bad data downstream (a standard real-world ETL expectation, and a differentiator vs. the
      rest of the portfolio, which doesn't have this anywhere) — `etl/validate.py` (Great
      Expectations, ephemeral context), raises `DataQualityError` with the full failure report
- [x] **Load**: load the processed/validated data into a queryable warehouse table (Postgres —
      RDS in the AWS deployment, plain Postgres via docker-compose for local dev), not just
      flat files — `etl/load.py`, one table per dataset plus an `etl_runs` registry table
- [x] Orchestrate extract → transform → validate → load as an **Airflow DAG** (same
      thin-`PythonOperator`-around-shared-code pattern already proven in the Financial Anomaly
      Detection Using RAG project's DAG — reuse that structure, don't reinvent it) —
      `airflow/dags/datoscope_etl_dag.py`, verified with `airflow dags test` against real
      MinIO/Postgres services (docker-compose)

## 2. API layer
- [x] Expose EDA/preprocessing/modeling/clustering as REST endpoints, reading from the warehouse
      table instead of local files — `api/` (FastAPI): `datasets` (ingest via the ETL pipeline:
      generate/upload/kaggle + clean, list, data), `eda`, `modeling` (regression/classification +
      model download), `clustering` (+ 2D projection), `comparison` (winner-selection, shared with
      `pages/4_Comparison.py` via the new `utils/comparison.py`). Verified end-to-end with real
      HTTP requests against live MinIO/Postgres for every endpoint, including the data-quality
      failure path (422) and 404s for unknown datasets.
- [x] Decouple the FastAPI backend from the Streamlit UI — Streamlit becomes one client, not the
      whole app. `utils/api_client.py` + sidebar (`utils/data_input.py`) and all four pages
      rewired to call the API over HTTP instead of `utils.generators`/`utils.preprocessing`/
      `utils.modeling` directly. Found and fixed a real bug uncovered by this rewire (transform
      reused the raw extract's run_id, so re-cleaning collided in the warehouse — see log.md), and
      added `test_dataset_name`/split support the original API design had missed. Verified in an
      actual browser (Playwright) end to end: generate+split → clean → EDA (all 7 endpoints) →
      classification training incl. the Random Forest tree view (downloads and unpickles the real
      model) → comparison → clustering incl. dendrogram/elbow diagnostics — zero console errors
      throughout.

## 3. Testing
- [x] pytest suite for `utils/preprocessing.py`, `utils/modeling.py`, `utils/generators.py`, and
      the new ETL transform/validation functions — pure-function-heavy, easy wins — `tests/`, 107
      tests total (98 pure-function unit tests with no external services — S3 mocked via `moto`,
      Great Expectations runs in-process — plus 9 `@pytest.mark.integration` tests against the
      real docker-compose warehouse for `etl/load.py`/`etl/pipeline.py`, verified passing against
      live services with automatic cleanup of test data). Found and fixed a real bug along the
      way: `gen_classification`'s "Imbalanced Classes" path was completely broken on the installed
      sklearn version (tuple `weights` vs. the `.copy()` call `make_classification` needs) — see
      log.md.
- [x] A data-quality test layer: assert the Great Expectations/Pandera checks actually catch bad
      data (feed known-bad fixtures through the transform stage) — `tests/test_etl_validate.py`:
      missing required columns, excess nulls, empty tables, and out-of-range values all confirmed
      to raise `DataQualityError` with the correct failure report, not just that clean data passes

## 4. MLOps
- [x] Docker/docker-compose for FastAPI + Streamlit + local Postgres (dev stack) — `Dockerfile`
      (shared by the new `api`/`streamlit` docker-compose services, different `command:` each);
      `docker compose up -d --build` now brings up the entire stack (MinIO, warehouse, MLflow,
      Airflow, API, Streamlit) in one command
- [x] **MLflow** — log every model-comparison run (existing `pages/4_Comparison.py` already
      computes R²/accuracy/Fowlkes-Mallows/Rand Index per model) as MLflow runs with
      params/metrics/artifacts; promote the best model per dataset-type to the **MLflow Model
      Registry** (staging → production stages) rather than just logging runs — `api/tracking.py`
      + `Dockerfile.mlflow`/`docker/mlflow_entrypoint.py` (tracking server backed by a new
      `mlflow` database in the warehouse Postgres + a new `datoscope-mlflow` MinIO bucket, both
      auto-created — no new infrastructure). Every regression/classification/clustering model
      trained via the API is logged as its own run; `/comparison/*` registers the winner and
      stages it automatically when called with `dataset_name`; `POST /models/{name}/promote`
      moves Staging → Production. Verified end-to-end against live services for all three task
      types, including the promotion/versions endpoints and their 404 error paths.
- [x] **DVC** for the processed datasets and model artifacts, with an **S3 remote** — this is the
      natural bridge between DVC and the AWS story below, not a separate concern. Scoped to
      **model artifacts only** (processed data lives in the warehouse, not flat files, per section
      1 — see log.md): MinIO remote (`datoscope-models` bucket), `scripts/
      05_export_production_model.py` pulls a registered model's Production version out of MLflow
      and writes it in the same payload shape the API's model-download endpoint uses. `dvc add` /
      `dvc push` / `dvc pull` verified end-to-end against live MinIO (round-tripped a real
      Production model through delete-and-restore).
- [x] **Git LFS** for large binary files that aren't model artifacts (those stay DVC's job) — the
      course report PDF and any future `.parquet`/`.pt`/`.h5`/etc. (`.gitattributes`). Set up
      forward-only (re-added the already-committed PDF under LFS as a new commit) rather than
      rewriting git history to retroactively migrate it, which would need a force-push.
- [x] **CI/CD** (GitHub Actions): CI job = lint + pytest on push; CD job = build the Docker image,
      push to **ECR**, then deploy to the Kubernetes cluster (see below) — one workflow, two jobs.
      CI done: `.github/workflows/ci.yml` (ruff + pytest, integration tests self-skip with no
      services running). **CD deliberately deferred** — it needs the ECR repo and K8s cluster from
      sections 5–6, which don't exist yet; building it now would mean either a non-functional stub
      or infrastructure decisions made out of order. Revisit once sections 5–6 are done.

## 5. AWS
- [x] **S3**: raw/processed/models/mlflow buckets (feeding the ETL pipeline above, plus the DVC
      remote for datasets/models) — `terraform/modules/s3` (versioning + SSE + public-access-block
      on all four, globally-unique names via a random suffix)
- [x] **RDS (Postgres)**: the warehouse table the FastAPI backend reads from in the deployed
      environment (docker-compose Postgres for local dev, as above) — `terraform/modules/rds`
      (db.t4g.micro / 20GB, free-tier-eligible, defaults to the account's default VPC)
- [x] **ECR**: container registry for the FastAPI/Streamlit/Airflow images — `terraform/modules/ecr`
      (image scanning on push, lifecycle policy expiring untagged images after 1 day / keeping the
      last 10 tagged)
- [x] **IAM**: least-privilege roles for the ETL job and the API service (S3 read/write scoped to
      the specific buckets/prefixes they need, not a broad admin role) — `terraform/modules/iam`
      (separate ETL/API roles, EKS-IRSA-ready via optional OIDC variables that default to a
      placeholder account-root trust until section 6's cluster exists)
- [x] **Terraform** for the above (S3 buckets, ECR repo, RDS instance, IAM roles) — infrastructure
      as code is itself a distinct, valued skill here, separate from just "using AWS services".
      Root config wires all four modules together (`terraform/main.tf`); `terraform validate`
      passes end-to-end. Went a step further than validate-only: `terraform/envs/localstack/` is a
      separate root module pointed at a local LocalStack container instead of real AWS — ran a real
      `terraform apply` for the `s3`+`iam` modules (21 resources), verified the actual buckets/
      roles/policies exist via the AWS CLI against LocalStack, then `terraform destroy`. `ecr` is
      Pro-only in LocalStack community edition (confirmed via a direct API call) and `rds` was
      skipped as flaky there, so both are validate-only — no real AWS credentials exist in this
      environment/`.env`, so an actual `terraform apply` against real AWS was not attempted.
- [x] Cost note: keep the live AWS footprint minimal — free-tier RDS, S3 (near-zero at this scale),
      spin up EKS only when demoing/screenshotting rather than leaving a cluster running; local
      dev/demo runs on `kind`/`minikube` against the same manifests (see below). Codified as
      defaults in `terraform/variables.tf` (`db.t4g.micro`, 20GB storage) rather than just a note.

## 6. Kubernetes
- [ ] Deployment/Service/Ingress manifests (or a Helm chart) for the FastAPI backend and Streamlit
      UI, with ConfigMaps for config and Secrets for credentials/API keys
- [ ] Deploy **Airflow itself on Kubernetes** via the official Helm chart with the
      `KubernetesExecutor` — a stronger, more realistic data-engineering story than running
      Airflow standalone, and reuses the DAG from section 1 as-is
- [ ] Document both paths: `kind`/`minikube` for local/free demo, and the real EKS deploy path
      (using the Terraform + ECR images from section 5) for when you actually want to show it live

## 7. AI/agentic layer (the differentiator — not just "added MLOps to a student app")
**Reordered ahead of sections 5–6 (AWS/Kubernetes)** at the user's request — no technical
dependency requires cloud infra first; the AI layer runs entirely against the local services
already built in sections 1–4. Runs on **Groq** (free tier, `GROQ_API_KEY`) rather than
OpenAI/Anthropic — see log.md for why.
- [x] LLM co-pilot that explains EDA findings and recommends preprocessing steps — a light RAG
      layer over sklearn/statistics documentation for grounded explanations — `agent/copilot.py` +
      `agent/rag.py`, corpus built from *installed* sklearn/scipy docstrings
      (`scripts/06_build_rag_corpus.py`, always in sync with pinned versions, no scraping).
      Citations are inline `[S1]` markers, deterministically verified against the actual retrieved
      source text (not just a prompt instruction) — `POST /copilot/{name}/explain|recommend`.
- [x] An agent that can autonomously run the extract → transform → validate → load → EDA → model
      → compare pipeline end-to-end from a natural-language goal (e.g. "find the best classifier
      for this churn dataset") and produce a written report — `agent/pipeline_agent.py`
      (`POST /agent/run`), a Groq tool-calling loop over `agent/tools.py`. Verified end-to-end
      against live services for real, including finding and fixing a genuine hallucination bug
      along the way (see log.md) — the agent now flags `had_tool_errors` deterministically from
      the actual tool trace rather than only trusting its own narrative.
- [x] Expose the pipeline as an **MCP server** so Claude Desktop (or the Patent Prior-Art agent's
      tooling) can drive DatoScope's analysis tools directly — `agent/mcp_server.py`, same
      `agent/tools.py` implementations the autonomous agent uses (11 tools total, including the
      co-pilot and the agent itself). Verified over the real stdio MCP protocol (initialize,
      list_tools, call_tool), not just that the module imports.

## 8. Tracing / observability
- [ ] **Langfuse** for the LLM co-pilot and autonomous-agent layer specifically (sessions,
      generations, scores) — this is the right tool for that half of the system, not generic spans
- [ ] Structured logs / OpenTelemetry for the non-LLM pipeline stages (ETL tasks, model training) —
      Langfuse and OTel covering their respective halves, not one tool stretched over both
- [x] **Prometheus + Grafana** for the metrics/dashboards half (request rate, p95 latency, error
      rate, status-code breakdown for the FastAPI backend) — `prometheus-fastapi-instrumentator`
      exposes `/metrics` on the API (`api/main.py`), scraped by the new `prometheus` docker-compose
      service (`observability/prometheus/prometheus.yml`), visualized by the new `grafana` service
      via file-based provisioning (`observability/grafana/provisioning/`, one dashboard —
      `datoscope-api.json` — 4 panels). Verified end-to-end against live containers: Prometheus
      target `up=1` for `job="datoscope-api"`, real `http_requests_total` series returned from a
      live query, Grafana's `/api/datasources` and `/api/search` confirm the Prometheus datasource
      and dashboard are both provisioned and loaded, not just files sitting unread in a mount.

## Why this version, not the original AutoML-only plan
Adding a real ETL pipeline + AWS + Kubernetes turns this from "a student Streamlit app with
MLOps bolted on" into a project that reads as Data Engineering as much as Data Science — broadens
which roles on the Data Science CV this project actually speaks to, without diluting the original
AI-co-pilot/MCP differentiator (section 7 is unchanged from the original plan, just renumbered).
