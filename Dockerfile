# Use a base image with the desired Python version
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
    && apt-get clean

# Install Poetry and configure it
RUN pip install poetry \
    && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-interaction --no-root

# Install PyTorch CPU-only
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

ENV HF_HOME=/tmp/ \
    TORCH_HOME=/tmp/ \
    OMP_NUM_THREADS=4

RUN python -c 'from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline; artifacts_path = StandardPdfPipeline.download_models_hf(force=True);'

# Pre-download EasyOCR models in compatible groups
RUN python -c 'import easyocr; \
    reader = easyocr.Reader(["fr", "de", "es", "en", "it", "pt"], gpu=False); \
    print("EasyOCR models downloaded successfully")'

COPY . .

# Railway provides PORT environment variable
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "poetry run uvicorn --port ${PORT:-8080} --host 0.0.0.0 main:app"]
