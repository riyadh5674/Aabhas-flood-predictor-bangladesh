# Aabhas staging preview

This staging image serves the built web app and FastAPI API from one origin. It
bundles the frozen D16 model and the feature-store snapshot present when the
image is built. The current workspace snapshot contains about 275 MB of
partitioned features through 2026-10-29; future local pipeline runs do not
update an already deployed image.

The image always displays a **Preview environment** banner and requires HTTP
Basic authentication when `AABHAS_PREVIEW_MODE=true`. Set
`AABHAS_PREVIEW_USER` and `AABHAS_PREVIEW_PASSWORD` as deployment secrets; the
application refuses to start in preview mode if either is missing. Use unique
credentials and HTTPS. Do not use this preview as an official warning service.

## Deploy to Google Cloud Run

Cloud Run can build from this repository's Dockerfile with `gcloud run deploy
--source .`; it sends source to Cloud Build and stores the image in Artifact
Registry. See Google's [source deployment guide](https://docs.cloud.google.com/run/docs/deploying-source-code).

Prerequisites:

1. Create/select a Google Cloud project and enable Cloud Run, Cloud Build, and
   Secret Manager. Billing may be required by the selected project configuration.
2. Install and authenticate the Google Cloud CLI, select the project, and create
   two Secret Manager secrets named `aabhas-preview-user` and
   `aabhas-preview-password`. Use a unique preview username and password.
3. Run the deploy command below from the repository root. Grant the Cloud Run
   service identity access to the two secrets when prompted or in IAM.

```powershell
gcloud run deploy aabhas-staging `
  --source . `
  --region asia-south1 `
  --allow-unauthenticated `
  --memory 2Gi `
  --cpu 1 `
  --max 2 `
  --min 0 `
  --set-env-vars AABHAS_PREVIEW_MODE=true `
  --set-secrets AABHAS_PREVIEW_USER=aabhas-preview-user:latest,AABHAS_PREVIEW_PASSWORD=aabhas-preview-password:latest
```

Cloud Run's unauthenticated invocation flag is needed so the browser can load
the SPA; the app-level Basic Auth gate protects both the page and API routes.
Keep the secret values out of source control and shell history. After deployment,
check `/health`, confirm the preview banner, verify an API forecast, and check
that unauthenticated requests receive HTTP 401.

## Current environment status

- No Git remote is configured in this workspace.
- A Google Cloud project has been created for staging: `aabhas-staging-20261030`.
  Project creation did not link a billing account.
- Docker and `gcloud` are not installed here, and the local CLI is not
  authenticated. Therefore the image has not been built and no hosted URL exists.
- Deployment is paused until billing/cost approval. Google Cloud requires an
  active billing account linked to a project to run Cloud services, including
  services whose usage may fall within free-tier allowances.
- The local Windows pipeline task does not update the feature snapshot inside a
  deployed container. Updating staging data requires a new image deployment or
  a separately designed external data store.
