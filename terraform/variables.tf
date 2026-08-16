variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment name, used as a resource-name/tag suffix"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Short project name, used as a resource-name prefix"
  type        = string
  default     = "datoscope"
}

# --- RDS -----------------------------------------------------------------

variable "warehouse_db_name" {
  description = "Initial database name on the warehouse RDS instance"
  type        = string
  default     = "datoscope"
}

variable "warehouse_db_username" {
  description = "Master username for the warehouse RDS instance"
  type        = string
  default     = "datoscope"
}

variable "warehouse_db_password" {
  description = "Master password for the warehouse RDS instance. Pass via TF_VAR_warehouse_db_password or a .tfvars file that's gitignored — never commit this."
  type        = string
  sensitive   = true
}

variable "rds_instance_class" {
  description = "RDS instance class — db.t4g.micro is free-tier eligible, kept as the default per the cost note in todo.md section 5"
  type        = string
  default     = "db.t4g.micro"
}

variable "rds_allocated_storage_gb" {
  description = "RDS allocated storage in GB — 20GB is the free-tier ceiling"
  type        = number
  default     = 20
}

# --- IAM / EKS IRSA (optional — only needed once section 6's EKS cluster exists) ---

variable "eks_oidc_provider_arn" {
  description = "EKS cluster's OIDC provider ARN, for IRSA trust policies. Leave null until the cluster exists (section 6) — the IAM roles are still created, just not yet assumable by any pod."
  type        = string
  default     = null
}

variable "eks_oidc_provider_url" {
  description = "EKS cluster's OIDC provider URL (without the https:// prefix), for IRSA trust policy conditions. Leave null until the cluster exists."
  type        = string
  default     = null
}

variable "k8s_namespace" {
  description = "Kubernetes namespace the DatoScope service accounts live in (must match section 6's manifests)"
  type        = string
  default     = "datoscope"
}
