variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "s3_bucket_arns" {
  description = "ARNs of the S3 buckets the ETL/API roles need read-write access to"
  type        = list(string)
}

variable "eks_oidc_provider_arn" {
  description = "EKS OIDC provider ARN for IRSA trust policies. When null, roles are created with an account-root assume-role trust as a placeholder, since IAM roles cannot exist without a principal — swap this in once section 6's EKS cluster exists."
  type        = string
  default     = null
}

variable "eks_oidc_provider_url" {
  description = "EKS OIDC provider URL (no https:// prefix), required alongside eks_oidc_provider_arn for IRSA"
  type        = string
  default     = null
}

variable "k8s_namespace" {
  description = "K8s namespace whose service accounts may assume these roles via IRSA"
  type        = string
  default     = "datoscope"
}
