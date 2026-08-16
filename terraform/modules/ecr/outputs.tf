output "repository_urls" {
  description = "Map of logical name to full ECR repository URL, for docker push/CD"
  value       = { for k, v in aws_ecr_repository.this : k => v.repository_url }
}

output "repository_arns" {
  value = { for k, v in aws_ecr_repository.this : k => v.arn }
}
