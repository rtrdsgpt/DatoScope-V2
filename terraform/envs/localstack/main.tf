module "s3" {
  source = "../../modules/s3"

  project_name = "datoscope"
  environment  = "localstack-test"
}

module "iam" {
  source = "../../modules/iam"

  project_name   = "datoscope"
  environment    = "localstack-test"
  s3_bucket_arns = values(module.s3.bucket_arns)
}
