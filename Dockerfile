# Hugging Face Spaces Dockerfile for Aabhas Flood Risk API
# Base: Python 3.13 slim (matches local Python 3.13.7)
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

# HF Spaces expects the app on port 7860
ENV PORT=7860
ENV HF_HOME=/app/.cache/huggingface

# Expose port
EXPOSE 7860

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "system.api.app:app", "--host", "0.0.0.0", "--port", "7860"]