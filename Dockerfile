# Render.com Dockerfile for Aabhas Flood Predictor (API + Frontend)
# Multi-stage build: build frontend, then copy to Python runtime

# ============================================================
# Stage 1: Build React frontend
# ============================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/web

# Copy package files first for better caching
COPY system/web/package*.json ./
RUN npm ci

# Copy frontend source and build
COPY system/web/ ./
RUN npm run build

# ============================================================
# Stage 2: Python runtime with API + built frontend
# ============================================================
FROM python:3.13-slim

# System dependencies for LightGBM/XGBoost and parquet
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Install huggingface_hub for feature store access
RUN pip install --no-cache-dir huggingface_hub==0.26.2

# Copy the entire project (needed for sys.path imports)
COPY . .

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/web/dist ./system/web/dist

# Render.com expects the app on port 7860 (or PORT env var)
ENV PORT=7860
ENV HF_HOME=/app/.cache/huggingface

# Expose port
EXPOSE 7860

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "system.api.app:app", "--host", "0.0.0.0", "--port", "7860"]