import os
from io import BytesIO
from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, Query

from document_converter.schema import ConversionResult
from document_converter.service import (
    DocumentConverterService,
    DoclingDocumentConversion,
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    DEFAULT_CONVERSION_MODE,
)
from document_converter.utils import is_file_format_supported

router = APIRouter()

# Lazy initialization - only create converter when needed (not at import time)
_converter = None
_service = None

def get_converter():
    global _converter
    mode = os.getenv("CONVERSION_MODE", DEFAULT_CONVERSION_MODE).lower()
    # In worker-process mode, keep Docling out of the API process entirely.
    if mode == "process":
        return None
    if _converter is None:
        # Get idle timeout from environment variable (default: 5 minutes)
        idle_timeout = int(os.getenv("MODEL_IDLE_TIMEOUT_SECONDS", DEFAULT_IDLE_TIMEOUT_SECONDS))
        _converter = DoclingDocumentConversion(idle_timeout_seconds=idle_timeout)
    return _converter

def get_service():
    global _service
    if _service is None:
        _service = DocumentConverterService(document_converter=get_converter())
    return _service


# Document direct conversion endpoints
@router.post(
    '/documents/convert',
    response_model=ConversionResult,
    response_model_exclude_unset=True,
    description="Convert a single document synchronously",
)
async def convert_single_document(
    document: UploadFile = File(...),
    extract_tables_as_images: bool = False,
    include_images: bool = False,
    image_resolution_scale: int = Query(4, ge=1, le=4),
):
    file_bytes = await document.read()
    if not is_file_format_supported(file_bytes, document.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {document.filename}")

    return get_service().convert_document(
        (document.filename, BytesIO(file_bytes)),
        extract_tables=extract_tables_as_images,
        include_images=include_images,
        image_resolution_scale=image_resolution_scale,
    )


@router.post(
    '/documents/batch-convert',
    response_model=List[ConversionResult],
    response_model_exclude_unset=True,
    description="Convert multiple documents synchronously",
)
async def convert_multiple_documents(
    documents: List[UploadFile] = File(...),
    extract_tables_as_images: bool = False,
    include_images: bool = False,
    image_resolution_scale: int = Query(4, ge=1, le=4),
):
    doc_streams = []
    for document in documents:
        file_bytes = await document.read()
        if not is_file_format_supported(file_bytes, document.filename):
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {document.filename}")
        doc_streams.append((document.filename, BytesIO(file_bytes)))

    return get_service().convert_documents(
        doc_streams,
        extract_tables=extract_tables_as_images,
        include_images=include_images,
        image_resolution_scale=image_resolution_scale,
    )
