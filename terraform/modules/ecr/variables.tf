variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "repository_names" {
  description = "Short logical names for ECR repos, e.g. [\"api\", \"streamlit\", \"airflow\"] — prefixed with project_name"
  type        = list(string)
  default     = ["api", "streamlit", "airflow"]
}
