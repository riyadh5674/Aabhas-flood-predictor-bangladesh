#!/usr/bin/env bash
# deploy-pipeline.sh - Deploy Aabhas daily pipeline to Cloud Run Job
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCP project created with billing enabled
#   - APIs enabled: run.googleapis.com, cloudscheduler.googleapis.com, artifactregistry.googleapis.com
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

echo "Deploying Aabhas pipeline to Cloud Run Job..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "HF Dataset: $HF_DATASET"

# Set project
gcloud config set project "$PROJECT_ID"

# Build and push image
echo "Building pipeline image..."
gcloud builds submit --config cloudbuild.yaml \
    --substitutions=_HF_DATASET="$HF_DATASET",_HF_TOKEN="$HF_TOKEN" \
    --project="$PROJECT_ID"

echo "Pipeline deployed successfully!"
echo ""
echo "To run manually:"
echo "  gcloud run jobs execute aabhas-daily-pipeline --region=$REGION --project=$PROJECT_ID"
echo ""
echo "To view logs:"
echo "  gcloud run jobs logs aabhas-daily-pipeline --region=$REGION --project=$PROJECT_ID"