from contextlib import asynccontextmanager

import os
import ctypes

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from document_converter.route import router as document_converter_router, get_converter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing needed
    yield
    # Shutdown: cleanup converters to free memory
    converter = get_converter()
    if converter is not None and hasattr(converter, 'cleanup_all'):
        converter.cleanup_all()


app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def malloc_trim_middleware(request, call_next):
    response = await call_next(request)
    # Optional: try to return freed heap memory back to OS after requests.
    # This can help RSS drop closer to cold-start baseline over time.
    if os.getenv("ENABLE_MALLOC_TRIM", "0").lower() in ("1", "true", "yes", "y", "on"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
        except Exception:
            pass
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


app.include_router(document_converter_router, prefix="", tags=["document-converter"])
