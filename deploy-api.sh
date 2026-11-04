#!/usr/bin/env bash
# deploy-api.sh - Deploy Aabhas API to Cloud Run
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCP project created with billing enabled
#   - APIs enabled: run.googleapis.com, artifactregistry.googleapis.com, secretmanager.googleapis.com
#   - Service accounts created (see setup-gcp.sh)

set -euo pipefail

PROJECT_ID="${1:-}"
REGION="${2:-asia-south1}"
HF_DATASET="${3:-}"
HF_TOKEN="${4:-}"

if [[ -z "$PROJECT_ID" || -z "$HF_DATASET" || -z "$HF_TOKEN" ]]; then
    echo "Usage: $0 <PROJECT_ID> <REGION> <HF_DATASET> <HF_TOKEN>"
    echo "Example: $0 my-flood-project asia-south1 riyadh5674/aabhas-flood-features hf_xxx"
    exit 1
fi

echo "Deploying Aabhas API to Cloud Run..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "HF Dataset: $HF_DATASET"

# Set project
gcloud config set project "$PROJECT_ID"

# Store HF token in Secret Manager (if not already)
echo "Ensuring HF_TOKEN secret exists..."
echo -n "$HF_TOKEN" | gcloud secrets create hf-token --data-file=- --project="$PROJECT_ID" 2>/dev/null || \
echo -n "$HF_TOKEN" | gcloud secrets versions add hf-token --data-file=- --project="$PROJECT_ID"

# Build and deploy via Cloud Build
echo "Building and deploying via Cloud Build..."
gcloud builds submit --config cloudbuild-api.yaml \
    --substitutions=_HF_DATASET="$HF_DATASET" \
    --project="$PROJECT_ID"

echo ""
echo "API deployed successfully!"
echo ""
echo "Staging URL:"
gcloud run services describe aabhas-api-staging --region="$REGION" --format='value(status.url)' --project="$PROJECT_ID"
echo ""
echo "Production URL:"
gcloud run services describe aabhas-api --region="$REGION" --format='value(status.url)' --project="$PROJECT_ID"
echo ""
echo "To view logs:"
echo "  gcloud run services logs read aabhas-api --region=$REGION --project=$PROJECT_ID"