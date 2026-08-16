output "etl_role_arn" {
  value = aws_iam_role.etl.arn
}

output "api_role_arn" {
  value = aws_iam_role.api.arn
}

output "irsa_enabled" {
  description = "Whether roles were created with real IRSA trust (true) or the account-root placeholder (false)"
  value       = local.irsa_enabled
}
