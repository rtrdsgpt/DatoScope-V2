terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

data "aws_caller_identity" "current" {}

locals {
  irsa_enabled = var.eks_oidc_provider_arn != null && var.eks_oidc_provider_url != null

  # Placeholder trust (account root) until section 6's EKS cluster + OIDC
  # provider exist — a role can't be created without some principal.
  placeholder_trust = {
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
      Action    = "sts:AssumeRole"
    }]
  }

  service_accounts = {
    etl = "airflow-etl"
    api = "datoscope-api"
  }
}

data "aws_iam_policy_document" "irsa_trust" {
  for_each = local.irsa_enabled ? local.service_accounts : {}

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [var.eks_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${var.eks_oidc_provider_url}:sub"
      values   = ["system:serviceaccount:${var.k8s_namespace}:${each.value}"]
    }

    condition {
      test     = "StringEquals"
      variable = "${var.eks_oidc_provider_url}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "etl" {
  name               = "${var.project_name}-${var.environment}-etl"
  assume_role_policy = local.irsa_enabled ? data.aws_iam_policy_document.irsa_trust["etl"].json : jsonencode(local.placeholder_trust)
}

resource "aws_iam_role" "api" {
  name               = "${var.project_name}-${var.environment}-api"
  assume_role_policy = local.irsa_enabled ? data.aws_iam_policy_document.irsa_trust["api"].json : jsonencode(local.placeholder_trust)
}

# ETL role: read-write on all data-lake buckets (raw/processed/models/mlflow) —
# Airflow tasks and DVC push/pull need full read-write across the pipeline.
data "aws_iam_policy_document" "etl_s3" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = concat(var.s3_bucket_arns, [for arn in var.s3_bucket_arns : "${arn}/*"])
  }
}

resource "aws_iam_role_policy" "etl_s3" {
  name   = "${var.project_name}-${var.environment}-etl-s3"
  role   = aws_iam_role.etl.id
  policy = data.aws_iam_policy_document.etl_s3.json
}

# API role: read-only on raw/processed, read-write on models (so the API can
# also register/version artifacts it produces via the modeling endpoints).
data "aws_iam_policy_document" "api_s3" {
  statement {
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = concat(var.s3_bucket_arns, [for arn in var.s3_bucket_arns : "${arn}/*"])
  }

  statement {
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:DeleteObject"]
    resources = [for arn in var.s3_bucket_arns : "${arn}/*"]
  }
}

resource "aws_iam_role_policy" "api_s3" {
  name   = "${var.project_name}-${var.environment}-api-s3"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api_s3.json
}
