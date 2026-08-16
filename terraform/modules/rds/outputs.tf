output "endpoint" {
  description = "RDS connection endpoint (host:port)"
  value       = aws_db_instance.warehouse.endpoint
}

output "address" {
  description = "RDS host address, without port"
  value       = aws_db_instance.warehouse.address
}

output "db_name" {
  value = aws_db_instance.warehouse.db_name
}

output "security_group_id" {
  value = aws_security_group.rds.id
}
