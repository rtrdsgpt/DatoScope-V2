module "s3" {
  source = "./modules/s3"

  project_name = var.project_name
  environment  = var.environment
}

module "ecr" {
  source = "./modules/ecr"

  project_name = var.project_name
  environment  = var.environment
}

module "rds" {
  source = "./modules/rds"

  project_name         = var.project_name
  environment          = var.environment
  db_name              = var.warehouse_db_name
  db_username          = var.warehouse_db_username
  db_password          = var.warehouse_db_password
  instance_class       = var.rds_instance_class
  allocated_storage_gb = var.rds_allocated_storage_gb
}

module "iam" {
  source = "./modules/iam"

  project_name          = var.project_name
  environment           = var.environment
  s3_bucket_arns        = values(module.s3.bucket_arns)
  eks_oidc_provider_arn = var.eks_oidc_provider_arn
  eks_oidc_provider_url = var.eks_oidc_provider_url
  k8s_namespace         = var.k8s_namespace
}
