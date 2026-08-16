provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "datoscope-v2"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
