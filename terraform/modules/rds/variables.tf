variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "db_name" {
  type = string
}

variable "db_username" {
  type = string
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "instance_class" {
  type = string
}

variable "allocated_storage_gb" {
  type = number
}

variable "vpc_id" {
  description = "VPC to launch the RDS instance in. Defaults to the account's default VPC when null."
  type        = string
  default     = null
}

variable "subnet_ids" {
  description = "Subnet IDs for the DB subnet group. Defaults to the default VPC's subnets when null."
  type        = list(string)
  default     = null
}

variable "publicly_accessible" {
  description = "Whether the RDS instance gets a public IP. Kept true by default since section 5 targets a free-tier dev setup with no bastion/VPN, not a production network topology."
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to reach RDS on the Postgres port"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
