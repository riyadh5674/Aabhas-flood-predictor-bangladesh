#!/usr/bin/env bash
# setup-gcp.sh - Set up GCP resources for Aabhas pipeline and API
# Run once per project

set -euo pipefail

PROJECT_ID="${1:-}"
REGION="${2:-asia-south1}"

if [[ -z "$PROJECT_ID" ]]; then
    echo "Usage: $0 <PROJECT_ID> [REGION]"
    exit 1
fi

echo "Setting up GCP project: $PROJECT_ID"

# Enable required APIs
gcloud services enable \
    run.googleapis.com \
    cloudscheduler.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    --project="$PROJECT_ID"

# Create service account for the pipeline job
gcloud iam service-accounts create aabhas-pipeline \
    --display-name="Aabhas Pipeline Runner" \
    --project="$PROJECT_ID" || true

# Create service account for the API
gcloud iam service-accounts create aabhas-api \
    --display-name="Aabhas API Service" \
    --project="$PROJECT_ID" || true

# Create service account for Cloud Scheduler
gcloud iam service-accounts create aabhas-scheduler \
    --display-name="Aabhas Scheduler Trigger" \
    --project="$PROJECT_ID" || true

# Grant Cloud Run invoker role to scheduler for pipeline
gcloud run services add-iam-policy-binding aabhas-daily-pipeline \
    --region="$REGION" \
    --member="serviceAccount:aabhas-scheduler@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker" \
    --project="$PROJECT_ID" 2>/dev/null || true

# Grant Cloud Run invoker role to scheduler for API (if needed for health checks)
gcloud run services add-iam-policy-binding aabhas-api \
    --region="$REGION" \
    --member="serviceAccount:aabhas-scheduler@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker" \
    --project="$PROJECT_ID" 2>/dev/null || true

# Grant necessary permissions to pipeline service account
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:aabhas-pipeline@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker" || true

# Create secret for HF token (run this manually with your token)
echo ""
echo "Next steps:"
echo "1. Store HF_TOKEN in Secret Manager:"
echo "   echo -n 'hf_xxx' | gcloud secrets create hf-token --data-file=- --project=$PROJECT_ID"
echo ""
echo "2. Deploy the pipeline:"
echo "   ./deploy-pipeline.sh $PROJECT_ID $REGION YOUR_HF_DATASET hf_xxx"
echo ""
echo "3. Deploy the API:"
echo "   ./deploy-api.sh $PROJECT_ID $REGION YOUR_HF_DATASET hf_xxx"