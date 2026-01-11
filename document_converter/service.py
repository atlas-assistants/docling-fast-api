import base64
import gc
import logging
import os
import json
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, TYPE_CHECKING, Any
from threading import Lock, Timer

from fastapi import HTTPException

from document_converter.schema import ConversionResult, ImageData
from document_converter.utils import handle_csv_file

logging.basicConfig(level=logging.INFO)
IMAGE_RESOLUTION_SCALE = 4

# Default idle timeout: 5 minutes (300 seconds)
# Models will be unloaded after this period of inactivity
DEFAULT_IDLE_TIMEOUT_SECONDS = 300

# Conversion mode:
# - "inprocess": run Docling in the API process (fast warm calls, hard to reliably free RSS)
# - "process": run Docling in a short-lived worker subprocess (serverless-like memory reclaim)
DEFAULT_CONVERSION_MODE = "inprocess"
DEFAULT_WORKER_TIMEOUT_SECONDS = 300

if TYPE_CHECKING:
    # Heavy imports only for type-checking. Runtime imports happen lazily in methods.
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # pragma: no cover
    from docling.document_converter import DocumentConverter  # pragma: no cover


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

    def __init__(self, pipeline_options: "PdfPipelineOptions" = None, idle_timeout_seconds: int = DEFAULT_IDLE_TIMEOUT_SECONDS):
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

    def _get_doc_converter(self, extract_tables: bool, include_images: bool, image_resolution_scale: int) -> "DocumentConverter":
        """Get or create a DocumentConverter instance with request-specific options."""
        # Lazy imports to keep API process light when CONVERSION_MODE=process
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import PdfFormatOption, DocumentConverter

        # Create a cache key based on the variable options
        # Note: include_images affects whether Docling generates image objects at all.
        cache_key = (extract_tables, include_images, image_resolution_scale)
        current_time = time.time()
        
        # Check if we already have a converter for these options
        if cache_key in self._doc_converters:
            # Update last used time
            self._last_used[cache_key] = current_time
            return self._doc_converters[cache_key]
        
        # Create pipeline options for this specific request
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = self.pipeline_options.generate_page_images
        # Avoid generating picture/table images unless explicitly requested.
        pipeline_options.generate_picture_images = bool(include_images)
        pipeline_options.ocr_options = self.pipeline_options.ocr_options
        pipeline_options.images_scale = image_resolution_scale
        pipeline_options.generate_table_images = bool(include_images and extract_tables)
        
        # Create converter with thread-safety
        # The ML models are loaded once and cached globally, so even if we create
        # a new DocumentConverter, the models won't be reloaded into memory.
        with self._lock:
            # Double-check after acquiring lock
            if cache_key not in self._doc_converters:
                memory_before = self._get_memory_usage_mb()
                logging.info(f"Loading ML models for converter (cache_key: {cache_key}). Memory before: {memory_before:.1f}MB")
                self._doc_converters[cache_key] = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                )
                self._last_used[cache_key] = current_time
                memory_after = self._get_memory_usage_mb()
                memory_used = memory_after - memory_before
                logging.info(f"ML models loaded into memory. Memory after: {memory_after:.1f}MB (used: {memory_used:.1f}MB)")
        
        return self._doc_converters[cache_key]
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except (ImportError, Exception):
            # Fallback: try to read from /proc/self/status on Linux
            try:
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            return float(line.split()[1]) / 1024  # Convert KB to MB
            except (FileNotFoundError, Exception):
                pass
            return 0.0
    
    def _cleanup_idle_converters(self):
        """Remove converters that have been idle for longer than the timeout."""
        current_time = time.time()
        idle_keys = []
        memory_before = self._get_memory_usage_mb()
        
        with self._lock:
            for cache_key, last_used in list(self._last_used.items()):
                idle_duration = current_time - last_used
                if idle_duration > self._idle_timeout:
                    idle_keys.append(cache_key)
            
            # Remove idle converters and try to explicitly clean up their internal state
            for cache_key in idle_keys:
                if cache_key in self._doc_converters:
                    converter = self._doc_converters[cache_key]
                    idle_duration = current_time - self._last_used[cache_key]
                    
                    # Try to explicitly clean up converter's internal state
                    try:
                        # Docling converters may have internal pipeline objects
                        if hasattr(converter, '_pipelines'):
                            converter._pipelines.clear()
                        if hasattr(converter, '_format_options'):
                            # Clear format options which may hold model references
                            for fmt_option in converter._format_options.values():
                                if hasattr(fmt_option, 'pipeline_options'):
                                    pipeline_opts = fmt_option.pipeline_options
                                    # Try to clear OCR reader if it exists
                                    if hasattr(pipeline_opts, 'ocr_options') and hasattr(pipeline_opts.ocr_options, 'reader'):
                                        try:
                                            del pipeline_opts.ocr_options.reader
                                        except (AttributeError, Exception):
                                            pass
                    except Exception as e:
                        logging.debug(f"Error cleaning converter internals: {e}")
                    
                    logging.info(f"Unloading idle converter (cache_key: {cache_key}, idle for {idle_duration:.1f}s)")
                    del self._doc_converters[cache_key]
                    del self._last_used[cache_key]
                    # Explicitly delete the converter object
                    del converter
        
        # Force garbage collection to help free memory
        if idle_keys:
            # Multiple GC passes to ensure cleanup
            for _ in range(5):  # Increased from 3 to 5
                collected = gc.collect()
                if collected == 0:
                    break  # No more objects to collect
            
            # Try to clear PyTorch cache if available
            try:
                import torch
                # Clear CPU cache (we're using CPU-only PyTorch)
                if hasattr(torch, 'empty_cache'):
                    torch.empty_cache()
                # Also try CUDA cache in case it's available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    # Force synchronization
                    torch.cuda.synchronize()
            except ImportError:
                pass
            except Exception as e:
                logging.debug(f"Could not clear PyTorch cache: {e}")
            
            # Try to clear EasyOCR global caches
            # EasyOCR caches Reader objects which hold models in memory
            try:
                import easyocr
                # EasyOCR may cache readers globally - try to clear module-level caches
                if hasattr(easyocr, '__dict__'):
                    # Clear any cached readers in the module
                    keys_to_remove = []
                    for key in easyocr.__dict__.keys():
                        if 'reader' in key.lower() or 'cache' in key.lower() or 'model' in key.lower():
                            keys_to_remove.append(key)
                    for key in keys_to_remove:
                        try:
                            delattr(easyocr, key)
                        except (AttributeError, Exception):
                            pass
                
                # Also try to clear any cached readers in submodules
                try:
                    import easyocr.reader
                    if hasattr(easyocr.reader, '__dict__'):
                        for key in list(easyocr.reader.__dict__.keys()):
                            if 'cache' in key.lower() or 'model' in key.lower():
                                try:
                                    delattr(easyocr.reader, key)
                                except (AttributeError, Exception):
                                    pass
                except (ImportError, AttributeError, Exception):
                    pass
            except (ImportError, Exception) as e:
                logging.debug(f"Could not clear EasyOCR cache: {e}")
            
            # Try to clear HuggingFace cache
            try:
                import transformers
                # Clear transformers cache if possible
                if hasattr(transformers, 'modeling_utils'):
                    # Force clear any cached models
                    pass
            except (ImportError, Exception):
                pass
            
            memory_after = self._get_memory_usage_mb()
            memory_freed = memory_before - memory_after
            logging.info(
                f"Unloaded {len(idle_keys)} idle converter(s). "
                f"Memory: {memory_before:.1f}MB -> {memory_after:.1f}MB "
                f"(freed: {memory_freed:.1f}MB)"
            )
        
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
        memory_before = self._get_memory_usage_mb()
        
        with self._lock:
            count = len(self._doc_converters)
            if count > 0:
                # Explicitly clean up each converter before deletion
                for cache_key, converter in list(self._doc_converters.items()):
                    try:
                        # Try to clean up converter internals
                        if hasattr(converter, '_pipelines'):
                            converter._pipelines.clear()
                        if hasattr(converter, '_format_options'):
                            for fmt_option in converter._format_options.values():
                                if hasattr(fmt_option, 'pipeline_options'):
                                    pipeline_opts = fmt_option.pipeline_options
                                    if hasattr(pipeline_opts, 'ocr_options') and hasattr(pipeline_opts.ocr_options, 'reader'):
                                        try:
                                            del pipeline_opts.ocr_options.reader
                                        except (AttributeError, Exception):
                                            pass
                    except Exception as e:
                        logging.debug(f"Error cleaning converter {cache_key}: {e}")
                
                logging.info(f"Force unloading {count} converter(s)")
                self._doc_converters.clear()
                self._last_used.clear()
            
            if self._cleanup_timer is not None:
                self._cleanup_timer.cancel()
                self._cleanup_timer = None
        
        # Force garbage collection (multiple passes)
        for _ in range(5):
            collected = gc.collect()
            if collected == 0:
                break
        
        # Clear PyTorch caches
        try:
            import torch
            if hasattr(torch, 'empty_cache'):
                torch.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
        except (ImportError, Exception):
            pass
        
        # Clear EasyOCR global caches
        try:
            import easyocr
            if hasattr(easyocr, '__dict__'):
                for key in list(easyocr.__dict__.keys()):
                    if 'reader' in key.lower() or 'cache' in key.lower():
                        try:
                            delattr(easyocr, key)
                        except (AttributeError, Exception):
                            pass
        except (ImportError, Exception):
            pass
        
        memory_after = self._get_memory_usage_mb()
        memory_freed = memory_before - memory_after
        logging.info(f"Cleanup complete. Memory: {memory_before:.1f}MB -> {memory_after:.1f}MB (freed: {memory_freed:.1f}MB)")

    def _update_pipeline_options(self, extract_tables: bool, image_resolution_scale: int) -> PdfPipelineOptions:
        self.pipeline_options.images_scale = image_resolution_scale
        self.pipeline_options.generate_table_images = extract_tables
        return self.pipeline_options

    @staticmethod
    def _setup_default_pipeline_options() -> "PdfPipelineOptions":
        # Lazy import
        from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True
        pipeline_options.ocr_options = EasyOcrOptions(lang=["fr", "de", "es", "en", "it", "pt"])

        return pipeline_options

    @staticmethod
    def _process_document_images(conv_res: Any, include_images: bool) -> Tuple[str, List[ImageData]]:
        # Lazy import
        from docling_core.types.doc import ImageRefMode, TableItem, PictureItem

        images = []
        table_counter = 0
        picture_counter = 0
        # Always export markdown with placeholders; we optionally embed images below.
        content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)

        if not include_images:
            return content_md, []

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
        include_images: bool = False,
        image_resolution_scale: int = IMAGE_RESOLUTION_SCALE,
    ) -> ConversionResult:
        # Lazy import
        from docling.datamodel.base_models import DocumentStream

        filename, file = document
        # Get converter (will load models if needed, or reuse existing)
        # Models will be automatically unloaded after idle timeout
        doc_converter = self._get_doc_converter(extract_tables, include_images, image_resolution_scale)

        if filename.lower().endswith('.csv'):
            file, error = handle_csv_file(file)
            if error:
                return ConversionResult(filename=filename, error=error)

        conv_res = doc_converter.convert(DocumentStream(name=filename, stream=file), raises_on_error=False)
        doc_filename = conv_res.input.file.stem

        if conv_res.errors:
            logging.error(f"Failed to convert {filename}: {conv_res.errors[0].error_message}")
            return ConversionResult(filename=doc_filename, error=conv_res.errors[0].error_message)

        content_md, images = self._process_document_images(conv_res, include_images=include_images)
        return ConversionResult(filename=doc_filename, markdown=content_md, images=images)

    def convert_batch(
        self,
        documents: List[Tuple[str, BytesIO]],
        extract_tables: bool = False,
        include_images: bool = False,
        image_resolution_scale: int = IMAGE_RESOLUTION_SCALE,
    ) -> List[ConversionResult]:
        # Lazy import
        from docling.datamodel.base_models import DocumentStream

        # Get converter (will load models if needed, or reuse existing)
        # Models will be automatically unloaded after idle timeout
        doc_converter = self._get_doc_converter(extract_tables, include_images, image_resolution_scale)

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

            content_md, images = self._process_document_images(conv_res, include_images=include_images)
            results.append(ConversionResult(filename=doc_filename, markdown=content_md, images=images))

        return results


class DocumentConverterService:
    def __init__(self, document_converter: Optional[DocumentConversionBase] = None):
        self.document_converter = document_converter

    def convert_document(self, document: Tuple[str, BytesIO], **kwargs) -> ConversionResult:
        mode = os.getenv("CONVERSION_MODE", DEFAULT_CONVERSION_MODE).lower()
        if mode == "process":
            return self._convert_document_via_worker(document, **kwargs)

        if self.document_converter is None:
            raise HTTPException(status_code=500, detail="Document converter not initialized")

        result = self.document_converter.convert(document, **kwargs)
        if result.error:
            logging.error(f"Failed to convert {document[0]}: {result.error}")
            raise HTTPException(status_code=500, detail=result.error)
        return result

    def convert_documents(self, documents: List[Tuple[str, BytesIO]], **kwargs) -> List[ConversionResult]:
        mode = os.getenv("CONVERSION_MODE", DEFAULT_CONVERSION_MODE).lower()
        if mode == "process":
            return self._convert_documents_via_worker(documents, **kwargs)

        if self.document_converter is None:
            raise HTTPException(status_code=500, detail="Document converter not initialized")

        return self.document_converter.convert_batch(documents, **kwargs)

    @staticmethod
    def _repo_root_dir() -> Path:
        # docling-fast-api/document_converter/service.py -> docling-fast-api/
        return Path(__file__).resolve().parents[1]

    def _run_worker(self, args: List[str]) -> str:
        timeout = int(os.getenv("WORKER_TIMEOUT_SECONDS", str(DEFAULT_WORKER_TIMEOUT_SECONDS)))
        cmd = [sys.executable, "-m", "document_converter.worker", *args]
        logging.info(f"CONVERSION_MODE=process: spawning worker subprocess: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            cwd=str(self._repo_root_dir()),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            # Include stderr to make debugging deploy issues easy
            raise HTTPException(
                status_code=500,
                detail=f"Worker failed (exit={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}",
            )
        logging.info("CONVERSION_MODE=process: worker subprocess completed successfully")
        return proc.stdout

    def _convert_document_via_worker(self, document: Tuple[str, BytesIO], **kwargs) -> ConversionResult:
        filename, fileobj = document
        extract_tables = bool(kwargs.get("extract_tables", False))
        include_images = bool(kwargs.get("include_images", False))
        image_resolution_scale = int(kwargs.get("image_resolution_scale", IMAGE_RESOLUTION_SCALE))

        tmp_path = None
        try:
            suffix = Path(filename).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                tmp_path = f.name
                fileobj.seek(0)
                f.write(fileobj.read())

            out = self._run_worker(
                [
                    "--mode",
                    "single",
                    "--input-path",
                    tmp_path,
                    "--filename",
                    filename,
                    "--extract-tables",
                    "true" if extract_tables else "false",
                    "--include-images",
                    "true" if include_images else "false",
                    "--image-scale",
                    str(image_resolution_scale),
                ]
            )
            payload = json.loads(out)
            return ConversionResult(**payload)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _convert_documents_via_worker(self, documents: List[Tuple[str, BytesIO]], **kwargs) -> List[ConversionResult]:
        extract_tables = bool(kwargs.get("extract_tables", False))
        include_images = bool(kwargs.get("include_images", False))
        image_resolution_scale = int(kwargs.get("image_resolution_scale", IMAGE_RESOLUTION_SCALE))

        tmp_dir = None
        batch_json_path = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="docling_batch_")
            batch = []

            for idx, (filename, fileobj) in enumerate(documents):
                suffix = Path(filename).suffix or ".bin"
                p = Path(tmp_dir) / f"input_{idx}{suffix}"
                fileobj.seek(0)
                p.write_bytes(fileobj.read())
                batch.append({"filename": filename, "path": str(p)})

            batch_json_path = str(Path(tmp_dir) / "batch.json")
            Path(batch_json_path).write_text(json.dumps(batch))

            out = self._run_worker(
                [
                    "--mode",
                    "batch",
                    "--batch-json-path",
                    batch_json_path,
                    "--extract-tables",
                    "true" if extract_tables else "false",
                    "--include-images",
                    "true" if include_images else "false",
                    "--image-scale",
                    str(image_resolution_scale),
                ]
            )
            payload = json.loads(out)
            return [ConversionResult(**item) for item in payload]
        finally:
            # Best-effort cleanup of temp files/dir
            if tmp_dir:
                try:
                    for child in Path(tmp_dir).glob("*"):
                        try:
                            child.unlink()
                        except OSError:
                            pass
                    Path(tmp_dir).rmdir()
                except OSError:
                    pass
