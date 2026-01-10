import base64
import gc
import logging
import time
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List, Tuple, Optional
from threading import Lock, Timer

from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import PdfFormatOption, DocumentConverter
from docling_core.types.doc import ImageRefMode, TableItem, PictureItem
from fastapi import HTTPException

from document_converter.schema import ConversionResult, ImageData
from document_converter.utils import handle_csv_file

logging.basicConfig(level=logging.INFO)
IMAGE_RESOLUTION_SCALE = 4

# Default idle timeout: 5 minutes (300 seconds)
# Models will be unloaded after this period of inactivity
DEFAULT_IDLE_TIMEOUT_SECONDS = 300


class DocumentConversionBase(ABC):
    @abstractmethod
    def convert(self, document: Tuple[str, BytesIO], **kwargs) -> ConversionResult:
        pass

    @abstractmethod
    def convert_batch(self, documents: List[Tuple[str, BytesIO]], **kwargs) -> List[ConversionResult]:
        pass


class DoclingDocumentConversion(DocumentConversionBase):
    """Document conversion implementation using Docling.

    You can initialize with default pipeline options or provide your own:

    Example:
        ```python
        # Using default options
        converter = DoclingDocumentConversion()

        # Or customize with your own pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = RapidOcrOptions()  # Use RapidOcrOptions instead of EasyOCR (note : you need to install the OCR package)
        pipeline_options.generate_page_images = True

        converter = DoclingDocumentConversion(pipeline_options=pipeline_options)
        ```
    """

    def __init__(self, pipeline_options: PdfPipelineOptions = None, idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS):
        self.pipeline_options = pipeline_options if pipeline_options else self._setup_default_pipeline_options()
        # Cache DocumentConverter instances by their option signature to reuse across requests
        # This prevents creating new converter instances on every request.
        # Models will be unloaded after idle_timeout_seconds of inactivity (serverless-like behavior)
        self._doc_converters = {}  # Cache keyed by (extract_tables, image_resolution_scale)
        self._last_used = {}  # Track last usage time for each converter
        self._lock = Lock()
        self._idle_timeout = idle_timeout_seconds
        self._cleanup_timer: Optional[Timer] = None
        self._schedule_cleanup()

    def _get_doc_converter(self, extract_tables: bool, image_resolution_scale: int) -> DocumentConverter:
        """Get or create a DocumentConverter instance with request-specific options."""
        # Create a cache key based on the variable options
        cache_key = (extract_tables, image_resolution_scale)
        current_time = time.time()
        
        # Check if we already have a converter for these options
        if cache_key in self._doc_converters:
            # Update last used time
            self._last_used[cache_key] = current_time
            return self._doc_converters[cache_key]
        
        # Create pipeline options for this specific request
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = self.pipeline_options.generate_page_images
        pipeline_options.generate_picture_images = self.pipeline_options.generate_picture_images
        pipeline_options.ocr_options = self.pipeline_options.ocr_options
        pipeline_options.images_scale = image_resolution_scale
        pipeline_options.generate_table_images = extract_tables
        
        # Create converter with thread-safety
        # The ML models are loaded once and cached globally, so even if we create
        # a new DocumentConverter, the models won't be reloaded into memory.
        with self._lock:
            # Double-check after acquiring lock
            if cache_key not in self._doc_converters:
                logging.info(f"Loading ML models for converter (cache_key: {cache_key})")
                self._doc_converters[cache_key] = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                )
                self._last_used[cache_key] = current_time
                logging.info("ML models loaded into memory")
        
        return self._doc_converters[cache_key]
    
    def _cleanup_idle_converters(self):
        """Remove converters that have been idle for longer than the timeout."""
        current_time = time.time()
        idle_keys = []
        
        with self._lock:
            for cache_key, last_used in list(self._last_used.items()):
                idle_duration = current_time - last_used
                if idle_duration > self._idle_timeout:
                    idle_keys.append(cache_key)
            
            # Remove idle converters
            for cache_key in idle_keys:
                if cache_key in self._doc_converters:
                    logging.info(f"Unloading idle converter (cache_key: {cache_key}, idle for {idle_duration:.1f}s)")
                    del self._doc_converters[cache_key]
                    del self._last_used[cache_key]
        
        # Force garbage collection to help free memory
        if idle_keys:
            gc.collect()
            # Try to clear PyTorch cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            except Exception as e:
                logging.debug(f"Could not clear PyTorch cache: {e}")
            
            logging.info(f"Unloaded {len(idle_keys)} idle converter(s). Memory should be freed.")
        
        # Schedule next cleanup check
        self._schedule_cleanup()
    
    def _schedule_cleanup(self):
        """Schedule the next cleanup check."""
        # Cancel existing timer if any
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
        
        # Schedule cleanup check every minute (or half the idle timeout, whichever is smaller)
        check_interval = min(60, self._idle_timeout / 2)
        self._cleanup_timer = Timer(check_interval, self._cleanup_idle_converters)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def cleanup_all(self):
        """Force cleanup of all converters (useful for shutdown or manual cleanup)."""
        with self._lock:
            count = len(self._doc_converters)
            if count > 0:
                logging.info(f"Force unloading {count} converter(s)")
                self._doc_converters.clear()
                self._last_used.clear()
            
            if self._cleanup_timer is not None:
                self._cleanup_timer.cancel()
                self._cleanup_timer = None
        
        # Force garbage collection
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, Exception):
            pass

    def _update_pipeline_options(self, extract_tables: bool, image_resolution_scale: int) -> PdfPipelineOptions:
        self.pipeline_options.images_scale = image_resolution_scale
        self.pipeline_options.generate_table_images = extract_tables
        return self.pipeline_options

    @staticmethod
    def _setup_default_pipeline_options() -> PdfPipelineOptions:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True
        pipeline_options.ocr_options = EasyOcrOptions(lang=["fr", "de", "es", "en", "it", "pt"])

        return pipeline_options

    @staticmethod
    def _process_document_images(conv_res) -> Tuple[str, List[ImageData]]:
        images = []
        table_counter = 0
        picture_counter = 0
        content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)

        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, (TableItem, PictureItem)) and element.image:
                img_buffer = BytesIO()
                element.image.pil_image.save(img_buffer, format="PNG")

                if isinstance(element, TableItem):
                    table_counter += 1
                    image_name = f"table-{table_counter}.png"
                    image_type = "table"
                else:
                    picture_counter += 1
                    image_name = f"picture-{picture_counter}.png"
                    image_type = "picture"
                    content_md = content_md.replace("<!-- image -->", image_name, 1)

                image_bytes = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                images.append(ImageData(type=image_type, filename=image_name, image=image_bytes))

        return content_md, images

    def convert(
        self,
        document: Tuple[str, BytesIO],
        extract_tables: bool = False,
        image_resolution_scale: int = IMAGE_RESOLUTION_SCALE,
    ) -> ConversionResult:
        filename, file = document
        # Get converter (will load models if needed, or reuse existing)
        # Models will be automatically unloaded after idle timeout
        doc_converter = self._get_doc_converter(extract_tables, image_resolution_scale)

        if filename.lower().endswith('.csv'):
            file, error = handle_csv_file(file)
            if error:
                return ConversionResult(filename=filename, error=error)

        conv_res = doc_converter.convert(DocumentStream(name=filename, stream=file), raises_on_error=False)
        doc_filename = conv_res.input.file.stem

        if conv_res.errors:
            logging.error(f"Failed to convert {filename}: {conv_res.errors[0].error_message}")
            return ConversionResult(filename=doc_filename, error=conv_res.errors[0].error_message)

        content_md, images = self._process_document_images(conv_res)
        return ConversionResult(filename=doc_filename, markdown=content_md, images=images)

    def convert_batch(
        self,
        documents: List[Tuple[str, BytesIO]],
        extract_tables: bool = False,
        image_resolution_scale: int = IMAGE_RESOLUTION_SCALE,
    ) -> List[ConversionResult]:
        # Get converter (will load models if needed, or reuse existing)
        # Models will be automatically unloaded after idle timeout
        doc_converter = self._get_doc_converter(extract_tables, image_resolution_scale)

        conv_results = doc_converter.convert_all(
            [DocumentStream(name=filename, stream=file) for filename, file in documents],
            raises_on_error=False,
        )

        results = []
        for conv_res in conv_results:
            doc_filename = conv_res.input.file.stem

            if conv_res.errors:
                logging.error(f"Failed to convert {conv_res.input.name}: {conv_res.errors[0].error_message}")
                results.append(ConversionResult(filename=conv_res.input.name, error=conv_res.errors[0].error_message))
                continue

            content_md, images = self._process_document_images(conv_res)
            results.append(ConversionResult(filename=doc_filename, markdown=content_md, images=images))

        return results


class DocumentConverterService:
    def __init__(self, document_converter: DocumentConversionBase):
        self.document_converter = document_converter

    def convert_document(self, document: Tuple[str, BytesIO], **kwargs) -> ConversionResult:
        result = self.document_converter.convert(document, **kwargs)
        if result.error:
            logging.error(f"Failed to convert {document[0]}: {result.error}")
            raise HTTPException(status_code=500, detail=result.error)
        return result

    def convert_documents(self, documents: List[Tuple[str, BytesIO]], **kwargs) -> List[ConversionResult]:
        return self.document_converter.convert_batch(documents, **kwargs)
