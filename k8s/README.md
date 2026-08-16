# Kubernetes (todo.md section 6)

Two things live here:

1. **`base/`** — Deployment/Service/Ingress + ConfigMap/Secret for the FastAPI backend and
   Streamlit UI (`api`/`streamlit` share one image, same as `docker-compose.yml`).
2. **`airflow/`** — Helm values to deploy Airflow itself on Kubernetes via the official chart,
   using the `KubernetesExecutor` (one pod per task, no Celery/worker fleet) — the same DAG from
   section 1 (`airflow/dags/datoscope_etl_dag.py`), now baked into `datoscope-airflow`'s image
   instead of bind-mounted.

Both paths below — local `kind` and real EKS — deploy the exact same manifests/chart; only the
image source and a few environment-specific values differ.

## Local path — kind

```bash
# 1. Cluster with ingress-nginx's expected node label + port mappings
kind create cluster --name datoscope --config k8s/kind-cluster.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl wait --namespace ingress-nginx --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller --timeout=180s

# 2. Build the app images (same Dockerfiles as docker-compose) and load into kind
docker build -f Dockerfile -t datoscope-app:local .
docker build -f Dockerfile.airflow -t datoscope-airflow:local .
kind load docker-image datoscope-app:local --name datoscope
kind load docker-image datoscope-airflow:local --name datoscope

# 3. API + Streamlit
cp k8s/base/secret.example.yaml k8s/base/secret.yaml   # fill in real values; gitignored
kubectl apply -k k8s/base/
kubectl -n datoscope wait --for=condition=available deployment --all --timeout=120s

curl http://localhost:8090/api/health          # -> {"status":"ok"}
curl http://localhost:8090/_stcore/health      # -> ok (Streamlit)

# 4. Airflow (metadata DB first, then the chart — see "Why plain postgres" below)
kubectl create namespace airflow
kubectl apply -f k8s/airflow/metadata-postgres.yaml
helm repo add apache-airflow https://airflow.apache.org
helm install datoscope-airflow apache-airflow/airflow \
  --version 1.15.0 \
  --namespace airflow \
  -f k8s/airflow/values-kind.yaml --timeout 10m
kubectl -n airflow port-forward svc/datoscope-airflow-webserver 8082:8080
# -> http://localhost:8082 (admin/admin, see values-kind.yaml note below)
```

**Chart version is pinned to `1.15.0`** — the repo's `latest`/unversioned chart now defaults to
targeting Airflow 3.x (renames `webserver` to `api-server`, changes several config keys), which
doesn't match this project's `apache/airflow:2.9.3` base image. `1.15.0` is the chart release whose
`appVersion` is `2.9.3` — confirmed via `helm search repo apache-airflow/airflow --versions`.

**Why plain `postgres:16` for Airflow's metadata DB, not the chart's bundled subchart**: the
chart's default Bitnami Postgres subchart pins `bitnami/postgresql:16.1.0-debian-11-r15`, which
Bitnami has since pulled from Docker Hub — `ImagePullBackOff` on a fresh install, not an
environment problem. `k8s/airflow/metadata-postgres.yaml` is the same `postgres:16` image already
used everywhere else in this project (`docker-compose.yml`'s `warehouse`/`airflow-metadata`
services), pointed to via the chart's `data.metadataConnection` values instead of `postgresql:
enabled: true`.

### What's verified vs. what isn't

- **API + Streamlit**: real `Deployment`/`Service`/`Ingress`, reached through the real
  ingress-nginx controller on a real kind cluster (`curl` through `localhost:8090`, not just
  `kubectl get pods`). Readiness/liveness probes hit each app's real health endpoint.
- **Airflow control plane**: scheduler + webserver reach `Running`/`Ready` with the
  `KubernetesExecutor` configured, backed by a real (if disposable) Postgres.
- **KubernetesExecutor task-pod launch**: triggering the section-1 DAG causes the scheduler to
  dynamically create a new pod per task (`kubectl -n airflow get pods -w` shows it appear and
  terminate) — the actual mechanism this todo item is about.
- **NOT verified**: the DAG's tasks completing successfully end-to-end. This kind cluster has no
  MinIO/warehouse Postgres deployed in it (those are section 5's S3/RDS in the real AWS target,
  and section 4's docker-compose services locally — deploying a second copy just for this smoke
  test would test the DAG's business logic, which sections 1/3/4 already cover, not the K8s/Helm
  wiring this section is actually about). So the launched task pod fails at its own connection
  step, which is expected and out of scope here — same "verify what's practical, document what
  isn't" split as the LocalStack ECR/RDS gap in section 5's log.md entry.

## EKS path (real AWS)

Same manifests/chart, different inputs:

```bash
# Cluster: not provisioned by this repo's Terraform yet (todo.md section 5 covers
# S3/RDS/ECR/IAM only, not an EKS module — creating/costing a real EKS cluster
# needs explicit go-ahead given the running-cluster cost). Once one exists:

aws eks update-kubeconfig --name <cluster-name> --region <region>

# Images come from ECR (terraform output ecr_repository_urls) instead of a
# locally-built/kind-loaded tag — see the `cd` job in .github/workflows/ci.yml,
# which builds+pushes+deploys automatically on every push to main.

kubectl apply -k k8s/base/          # after editing image: fields to the ECR URLs,
                                     # or via `kustomize edit set image` in CI
helm upgrade --install datoscope-airflow apache-airflow/airflow \
  --version 1.15.0 \
  --namespace airflow \
  -f k8s/airflow/values-eks.yaml \
  --set images.airflow.repository=<ecr-repo-url> \
  --set images.airflow.tag=<tag>
```

`values-eks.yaml` differs from `values-kind.yaml` in the ways that matter for a real deploy: real
ECR image (not a local tag), a real external Postgres for Airflow's metadata DB via
`data.metadataConnection`/`envFrom` (not the disposable in-cluster one), and an EKS-IRSA service
account annotation slot tied to `terraform/modules/iam`'s `etl` role once its
`eks_oidc_provider_arn`/`_url` are filled in. **Not applied against a real cluster** — no EKS
cluster exists in this environment, and creating one has a real hourly cost that wasn't
authorized. Reviewed for correctness against the actual Terraform outputs and this directory's
layout rather than left as an unreviewed stub.

## Cleanup

```bash
helm uninstall datoscope-airflow --namespace airflow
kubectl delete namespace airflow datoscope ingress-nginx
kind delete cluster --name datoscope
```
