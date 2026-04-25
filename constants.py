"""
Project-level constants shared across the pipeline.

Holds non-secret infrastructure identifiers (GCP project, GCS bucket, etc.)
that every collaborator and every pipeline scenario needs the same value for.
Checked into git on purpose so a fresh clone is immediately runnable.

================================================================================
DO NOT PUT SECRETS IN THIS FILE.
================================================================================

This file is committed to source control. Anything here is visible to anyone
with repo access (and to anyone the repo is ever shared with). Never add:

    - API keys / access tokens
    - GCP / AWS service-account JSON or private key material
    - OAuth client secrets
    - Database passwords or connection strings containing credentials
    - Any other value that grants access to a system

For secrets, use environment variables or a gitignored `.env` file loaded via
`python-dotenv` or your CI's secret-injection mechanism.
"""

PROJECT_ID = "maduros-dolce"
BUCKET_NAME = "maduros-dolce-capstone-data"
