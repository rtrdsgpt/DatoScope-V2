output "s3_bucket_names" {
  value = module.s3.bucket_names
}

output "ecr_repository_urls" {
  value = module.ecr.repository_urls
}

output "rds_endpoint" {
  value = module.rds.endpoint
}

output "etl_role_arn" {
  value = module.iam.etl_role_arn
}

output "api_role_arn" {
  value = module.iam.api_role_arn
}
