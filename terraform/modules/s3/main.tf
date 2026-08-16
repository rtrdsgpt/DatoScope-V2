terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
    random = {
      source = "hashicorp/random"
    }
  }
}

# S3 bucket names are globally unique across all of AWS, so a random suffix
# is appended rather than relying on project+environment staying collision-free.
resource "random_id" "suffix" {
  byte_length = 4
}

locals {
  bucket_names = {
    raw       = "${var.project_name}-${var.environment}-raw-${random_id.suffix.hex}"
    processed = "${var.project_name}-${var.environment}-processed-${random_id.suffix.hex}"
    models    = "${var.project_name}-${var.environment}-models-${random_id.suffix.hex}"
    mlflow    = "${var.project_name}-${var.environment}-mlflow-${random_id.suffix.hex}"
  }
}

resource "aws_s3_bucket" "this" {
  for_each = local.bucket_names

  bucket = each.value
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this

  bucket = each.value.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this

  bucket = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each = aws_s3_bucket.this

  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
