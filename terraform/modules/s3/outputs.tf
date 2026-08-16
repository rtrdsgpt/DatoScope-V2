output "bucket_names" {
  description = "Map of logical name (raw/processed/models/mlflow) to actual S3 bucket name"
  value       = { for k, v in aws_s3_bucket.this : k => v.bucket }
}

output "bucket_arns" {
  description = "Map of logical name to S3 bucket ARN, for IAM policy attachment"
  value       = { for k, v in aws_s3_bucket.this : k => v.arn }
}
