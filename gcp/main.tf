provider "google" {
  project = var.project_id
}

variable "project_id" {
  type = string
  sensitive = true
}

# 1. Create the Service Account
resource "google_service_account" "github_actions_sa" {
  account_id   = "github-actions-demo-sa"
  display_name = "GitHub Actions Demo Service Account"
}

# 2. Assign Required Roles (Least Privilege)
locals {
  roles = [
    "roles/artifactregistry.writer", # To push Docker images
    "roles/run.admin",               # To deploy/manage Cloud Run services
    "roles/iam.serviceAccountUser",  # To "act as" the runtime SA of Cloud Run
  ]
}

resource "google_project_iam_member" "github_actions_roles" {
  for_each = toset(local.roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.github_actions_sa.email}"
}

# 3. Generate the JSON Key
resource "google_service_account_key" "github_actions_key" {
  service_account_id = google_service_account.github_actions_sa.name
}

# 4. Output the Key (to be copied to GitHub Secrets)
output "service_account_key" {
  value     = google_service_account_key.github_actions_key.private_key
  sensitive = true
}
