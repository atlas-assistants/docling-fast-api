# Documents to Markdown Converter Server

> [!IMPORTANT]
> This backend server is a robust solution for effortlessly converting a wide range of document formats—including PDF, DOCX, PPTX, CSV, HTML, JPG, PNG, TIFF, BMP, AsciiDoc, and Markdown—into Markdown. Powered by [Docling](https://github.com/DS4SD/docling) (IBM's advanced document parser), this service is built with FastAPI, ensuring fast, efficient processing. Optimized for CPU-only mode, this solution offers high performance and flexibility, making it ideal for handling complex document processing.

## Features
- **Multiple Format Support**: Converts various document types including:
  - PDF files
  - Microsoft Word documents (DOCX)
  - PowerPoint presentations (PPTX)
  - HTML files
  - Images (JPG, PNG, TIFF, BMP)
  - AsciiDoc files
  - Markdown files
  - CSV files

- **Conversion Capabilities**:
  - Text extraction and formatting
  - Table detection, extraction and conversion
  - Image extraction and processing
  - Multi-language OCR support (French, German, Spanish, English, Italian, Portuguese etc)
  - Configurable image resolution scaling

- **API Endpoints**:
  - Synchronous single document conversion
  - Synchronous batch document conversion

## Environment Setup (Running Locally)

### Prerequisites
- Python 3.12 or higher
- Poetry (Python package manager)

### 1. Install Poetry (if not already installed)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone and Setup Project
```bash
git clone https://github.com/drmingler/docling-api.git
cd docling-api
poetry install
```

### 3. Start the Application

Start the FastAPI server:
```bash
poetry run uvicorn main:app --reload --port 8080
```

### 4. Verify Installation

Check if the API server is running:
```bash
curl http://localhost:8080/docs
```

Test the API:
```bash
curl -X POST "http://localhost:8080/documents/convert" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "document=@/path/to/test.pdf"
```

### Development Notes

- The API documentation is available at http://localhost:8080/docs
- The service supports synchronous document conversion
- For development, the server runs with auto-reload enabled

## Environment Setup (Running in Docker)

1. Clone the repository:
```bash
git clone https://github.com/drmingler/docling-api.git
cd docling-api
```

2. Start the service using Docker Compose:
```bash
docker-compose -f docker-compose.cpu.yml up --build
```

## Service Components

The service will start the following component:

- **API Server**: http://localhost:8080

## API Usage

### Single Document Conversion

Convert a single document immediately:

```bash
curl -X POST "http://localhost:8080/documents/convert" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "document=@/path/to/document.pdf" \
  -F "extract_tables_as_images=true" \
  -F "image_resolution_scale=4"
```

### Batch Processing

Convert multiple documents:

```bash
curl -X POST "http://localhost:8080/documents/batch-convert" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "documents=@/path/to/document1.pdf" \
  -F "documents=@/path/to/document2.pdf" \
  -F "extract_tables_as_images=true" \
  -F "image_resolution_scale=4"
```

## Configuration Options

- `image_resolution_scale`: Control the resolution of extracted images (1-4)
- `extract_tables_as_images`: Extract tables as images (true/false)
- `include_images`: Embed extracted images as base64 strings in the JSON response (true/false, default: false)

## Memory / Worker modes

Docling + OCR models can use multiple GB of RAM when loaded. You can choose how the server runs conversions:

- **`CONVERSION_MODE=inprocess`** (default): runs Docling inside the API process.
- **`CONVERSION_MODE=process`**: spawns a new worker subprocess per request (memory reclaimed when the worker exits).
- **`CONVERSION_MODE=pool`**: keeps **one** persistent worker subprocess and reuses it for all requests, then kills it after an idle timeout.

Related env vars:

- **`WORKER_TIMEOUT_SECONDS`**: max seconds for a conversion before failing (default: 300)
- **`WORKER_IDLE_TIMEOUT_SECONDS`**: only for `CONVERSION_MODE=pool`; kill the persistent worker after N seconds idle (default: 300)
- **`ENABLE_MALLOC_TRIM`**: `1` to attempt to trim the API process heap after requests (Linux/glibc only; optional)

## Architecture

The service uses a simple architecture with the following component:

1. FastAPI application serving the REST API
2. Docling for the file conversion

## License
The codebase is under MIT license. See LICENSE for more information

## Acknowledgements
- [Docling](https://github.com/DS4SD/docling) the state-of-the-art document conversion library by IBM
- [FastAPI](https://fastapi.tiangolo.com/) the web framework
