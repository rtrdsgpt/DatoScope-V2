# LocalStack verification environment

A separate root module — deliberately **not** the same provider config as `terraform/` — that
points the AWS provider at a local [LocalStack](https://localstack.cloud) container instead of
real AWS. Its only purpose is to let `terraform apply` run for real (real state, real create/
destroy plumbing, real cross-resource references) without needing AWS credentials or spending
money, as a stronger check than `terraform validate` alone.

Only the `s3` and `iam` modules are wired up here. `ecr` and `rds` are **not** — LocalStack's
free/community edition doesn't emulate ECR at all (confirmed: `DescribeRepositories` returns
`API for service 'ecr' not yet implemented or pro feature`) and its RDS support is limited/flaky
for a quick smoke test. Both of those modules are checked with `terraform validate` in the main
config only (see `../../README` / log.md).

## Usage

```bash
# 1. Start LocalStack (separate from the app's docker-compose.yml — this is a
#    test-only container, not part of the dev stack)
docker run -d --rm --name datoscope-localstack -p 4566:4566 localstack/localstack:latest

# wait for it to be healthy
until curl -s http://localhost:4566/_localstack/health | grep -q '"s3": "available"'; do sleep 2; done

# 2. Apply against it
cd terraform/envs/localstack
terraform init
terraform apply -auto-approve

# 3. Verify resources actually exist in LocalStack
aws --endpoint-url=http://localhost:4566 s3 ls
aws --endpoint-url=http://localhost:4566 iam list-roles

# 4. Tear down
terraform destroy -auto-approve
docker stop datoscope-localstack
```
