# Use a base image with the desired Python version
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies needed to compile C extensions
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        gcc \
        g++ \
        make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (including those requiring compilation)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi==0.115.4 uvicorn==0.32.0 docling==2.25.1 python-multipart==0.0.17 && \
    pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Final stage: copy installed packages to a clean image
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update \
    && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
        curl \
        wget \
        git \
        procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set environment variables for model caching
# Models will be downloaded on first use instead of during build
ENV HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    OMP_NUM_THREADS=4 \
    EASYOCR_MODULE_PATH=/app/.cache/easyocr

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch /app/.cache/easyocr

COPY . .

# Railway provides PORT environment variable
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn --port ${PORT:-8080} --host 0.0.0.0 --workers ${WEB_CONCURRENCY:-1} main:app"]
