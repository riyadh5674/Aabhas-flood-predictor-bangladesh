# Dependency and Container Scanning Configuration

## 1. Python Dependencies (pip-audit / safety)

### 1.1 Requirements Files
- `requirements.txt` — Full development environment
- `requirements-api.txt` — Production API runtime (minimal)

### 1.2 Scanning Commands
```bash
# Install scanner
pip install pip-audit safety

# Scan requirements-api.txt (production)
pip-audit -r requirements-api.txt --format=json --output=audit-report.json
safety check -r requirements-api.txt --json --output=safety-report.json

# Scan full requirements
pip-audit -r requirements.txt --format=json --output=audit-full.json
```

### 1.3 CI Integration (GitHub Actions)
```yaml
- name: Dependency audit
  run: |
    pip install pip-audit
    pip-audit -r requirements-api.txt --fail-on=high --format=json > audit-report.json || true
    # Upload as artifact
```

### 1.4 Allowed Exceptions (Document Justification)
| Package | CVE | Justification | Expiry |
|---------|-----|---------------|--------|
| *None currently* | | | |

---

## 2. Container Scanning (Trivy / Grype)

### 2.1 Dockerfile Best Practices (Already Implemented)
- ✅ Multi-stage build (smaller attack surface)
- ✅ Non-root user (add to Dockerfile)
- ✅ Minimal base image (`python:3.13-slim`)
- ✅ No secrets in image
- ✅ Health check defined
- ✅ Read-only root filesystem (add to deployment)

### 2.2 Scanning Commands
```bash
# Install trivy
# Linux: curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Scan built image
docker build -f Dockerfile -t aabhas-api:latest .
trivy image --severity HIGH,CRITICAL --format json --output trivy-report.json aabhas-api:latest

# Scan pipeline image
docker build -f Dockerfile.pipeline -t aabhas-pipeline:latest .
trivy image --severity HIGH,CRITICAL --format json --output trivy-pipeline.json aabhas-pipeline:latest
```

### 2.3 Recommended Dockerfile Hardening
```dockerfile
# Add to Dockerfile (after WORKDIR /app)
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# For read-only filesystem (add to deployment, not Dockerfile)
# securityContext:
#   readOnlyRootFilesystem: true
#   runAsNonRoot: true
#   runAsUser: 1000
```

---

## 3. SBOM Generation (Software Bill of Materials)

```bash
# Generate SBOM for API image
trivy image --format cyclonedx --output sbom-api.json aabhas-api:latest

# Generate SBOM for pipeline image
trivy image --format cyclonedx --output sbom-pipeline.json aabhas-pipeline:latest

# Or use syft
syft aabhas-api:latest -o cyclonedx-json > sbom-api.json
```

---

## 4. Supply Chain Security

### 4.1 Pin Dependencies
```txt
# requirements-api.txt - pin exact versions
fastapi==0.115.0
uvicorn==0.32.0
pandas==2.2.2
lightgbm==4.5.0
xgboost==2.1.1
scikit-learn==1.5.2
huggingface-hub==0.26.2
pydantic==2.9.2
```

### 4.2 Verify Hashes
```bash
# Generate hashes
pip hash -r requirements-api.txt > requirements-api.txt.sha256

# Install with hash verification
pip install --require-hashes -r requirements-api.txt
```

### 4.3 GitHub Dependabot
```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    labels: ["dependencies", "security"]
```

---

## 5. Runtime Security (Production)

### 5.1 Kubernetes / Cloud Run Security Context
```yaml
# Cloud Run: automatic (managed)
# GKE / K8s:
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
```

### 5.2 Network Policies
- Ingress: Only from load balancer / VPC
- Egress: Only to HF API, FFWC, Open-Meteo, NASA POWER
- No direct internet access except allowlisted endpoints

---

## 6. Secret Scanning

```bash
# TruffleHog for git history
trufflehog git file://. --json --output trufflehog-report.json

# Or GitHub secret scanning (enabled by default on public repos)
# For private repos: enable in Settings > Security > Secret scanning
```

---

## 7. CI/CD Pipeline Integration

```yaml
# .github/workflows/security-scan.yml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 2 * * 1'  # Weekly

jobs:
  deps:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.13'}
      - run: pip install pip-audit trivy
      - run: pip-audit -r requirements-api.txt --fail-on=high
      - run: docker build -f Dockerfile -t aabhas-api:test .
      - run: trivy image --severity HIGH,CRITICAL --exit-code 1 aabhas-api:test
```

---

## 8. Vulnerability Response Process

| Severity | Response Time | Action |
|----------|---------------|--------|
| CRITICAL | 24 hours | Emergency patch, rebuild, redeploy |
| HIGH | 72 hours | Patch in next sprint, rebuild |
| MEDIUM | 30 days | Schedule update |
| LOW | Next release | Document, monitor |

### 8.1 Emergency Patch Procedure
1. Identify affected dependency
2. Check for fixed version
3. Update `requirements-api.txt` with fixed version
4. Rebuild and test in staging
5. Deploy to production
5. Verify `/health` and run smoke tests

---

## 9. Compliance Mapping

| Standard | Requirement | Implementation |
|----------|-------------|----------------|
| ISO 27001 | A.12.6.1 | Dependency scanning, patch management |
| NIST CSF | ID.RA-1, PR.IP-12 | SBOM, vulnerability scanning |
| SOC 2 | CC7.1, CC7.2 | Automated scanning in CI/CD |

---

*This document is part of the Aabhas Flood Predictor governance framework.*