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
- [ ] Fork `tanmoyghosh704-lang/DatoScope` (default branch `mehak-work`) into your own repo,
      clone it into this folder
- [ ] Clean up on fork: remove committed `.DS_Store` files, fix stale README "Project Structure"
      section (references deleted `Preprocessing.py`), fix `.devcontainer/devcontainer.json`
      (still references deleted `Preprocessing.py` in `openFiles`/`postAttachCommand`)
- [ ] Add a license (currently `license: null`)

## 1. ETL pipeline (new — this is the data-engineering core)
- [ ] **Extract**: ingestion stage that lands raw data in an S3 "raw" zone — cover both paths
      DatoScope already supports (user-uploaded files, synthetically generated datasets), plus
      add one real external source (e.g. a public dataset API/Kaggle) so the extract step isn't
      purely synthetic
- [ ] **Transform**: refactor the existing `utils/preprocessing.py` logic into an explicit
      transform stage (cleaning, outlier handling, encoding, feature engineering) that reads from
      the raw zone and writes to a "processed" zone — not ad hoc in-app calls
- [ ] **Data quality validation** in the transform stage — schema/null/range checks via Great
      Expectations or Pandera; fail the pipeline loudly on violation rather than silently passing
      bad data downstream (a standard real-world ETL expectation, and a differentiator vs. the
      rest of the portfolio, which doesn't have this anywhere)
- [ ] **Load**: load the processed/validated data into a queryable warehouse table (Postgres —
      RDS in the AWS deployment, plain Postgres via docker-compose for local dev), not just
      flat files
- [ ] Orchestrate extract → transform → validate → load as an **Airflow DAG** (same
      thin-`PythonOperator`-around-shared-code pattern already proven in the Financial Anomaly
      Detection Using RAG project's DAG — reuse that structure, don't reinvent it)

## 2. API layer
- [ ] Decouple a FastAPI backend from the Streamlit UI — expose EDA/preprocessing/modeling/
      clustering as REST endpoints, reading from the warehouse table instead of local files;
      Streamlit becomes one client, not the whole app

## 3. Testing
- [ ] pytest suite for `utils/preprocessing.py`, `utils/modeling.py`, `utils/generators.py`, and
      the new ETL transform/validation functions — pure-function-heavy, easy wins
- [ ] A data-quality test layer: assert the Great Expectations/Pandera checks actually catch bad
      data (feed known-bad fixtures through the transform stage)

## 4. MLOps
- [ ] Docker/docker-compose for FastAPI + Streamlit + local Postgres (dev stack)
- [ ] **MLflow** — log every model-comparison run (existing `pages/4_Comparison.py` already
      computes R²/accuracy/Fowlkes-Mallows/Rand Index per model) as MLflow runs with
      params/metrics/artifacts; promote the best model per dataset-type to the **MLflow Model
      Registry** (staging → production stages) rather than just logging runs
- [ ] **DVC** for the processed datasets and model artifacts, with an **S3 remote** — this is the
      natural bridge between DVC and the AWS story below, not a separate concern
- [ ] **CI/CD** (GitHub Actions): CI job = lint + pytest on push; CD job = build the Docker image,
      push to **ECR**, then deploy to the Kubernetes cluster (see below) — one workflow, two jobs

## 5. AWS
- [ ] **S3**: raw and processed data-lake zones (feeding the ETL pipeline above), plus the DVC
      remote for datasets/models
- [ ] **RDS (Postgres)**: the warehouse table the FastAPI backend reads from in the deployed
      environment (docker-compose Postgres for local dev, as above)
- [ ] **ECR**: container registry for the FastAPI/Streamlit/Airflow images
- [ ] **IAM**: least-privilege roles for the ETL job and the API service (S3 read/write scoped to
      the specific buckets/prefixes they need, not a broad admin role)
- [ ] **Terraform** for the above (S3 buckets, ECR repo, RDS instance, IAM roles) — infrastructure
      as code is itself a distinct, valued skill here, separate from just "using AWS services"
- [ ] Cost note: keep the live AWS footprint minimal — free-tier RDS, S3 (near-zero at this scale),
      spin up EKS only when demoing/screenshotting rather than leaving a cluster running; local
      dev/demo runs on `kind`/`minikube` against the same manifests (see below)

## 6. Kubernetes
- [ ] Deployment/Service/Ingress manifests (or a Helm chart) for the FastAPI backend and Streamlit
      UI, with ConfigMaps for config and Secrets for credentials/API keys
- [ ] Deploy **Airflow itself on Kubernetes** via the official Helm chart with the
      `KubernetesExecutor` — a stronger, more realistic data-engineering story than running
      Airflow standalone, and reuses the DAG from section 1 as-is
- [ ] Document both paths: `kind`/`minikube` for local/free demo, and the real EKS deploy path
      (using the Terraform + ECR images from section 5) for when you actually want to show it live

## 7. AI/agentic layer (the differentiator — not just "added MLOps to a student app")
- [ ] LLM co-pilot that explains EDA findings and recommends preprocessing steps — a light RAG
      layer over sklearn/statistics documentation for grounded explanations
- [ ] An agent that can autonomously run the extract → transform → validate → load → EDA → model
      → compare pipeline end-to-end from a natural-language goal (e.g. "find the best classifier
      for this churn dataset") and produce a written report
- [ ] Expose the pipeline as an **MCP server** so Claude Desktop (or the Patent Prior-Art agent's
      tooling) can drive DatoScope's analysis tools directly

## 8. Tracing / observability
- [ ] **Langfuse** for the LLM co-pilot and autonomous-agent layer specifically (sessions,
      generations, scores) — this is the right tool for that half of the system, not generic spans
- [ ] Structured logs / OpenTelemetry for the non-LLM pipeline stages (ETL tasks, model training) —
      Langfuse and OTel covering their respective halves, not one tool stretched over both

## Why this version, not the original AutoML-only plan
Adding a real ETL pipeline + AWS + Kubernetes turns this from "a student Streamlit app with
MLOps bolted on" into a project that reads as Data Engineering as much as Data Science — broadens
which roles on the Data Science CV this project actually speaks to, without diluting the original
AI-co-pilot/MCP differentiator (section 7 is unchanged from the original plan, just renumbered).
