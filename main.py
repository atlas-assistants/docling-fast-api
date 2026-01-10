import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from document_converter.route import router as document_converter_router, get_converter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing needed
    yield
    # Shutdown: cleanup converters to free memory
    converter = get_converter()
    if hasattr(converter, 'cleanup_all'):
        converter.cleanup_all()


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


app.include_router(document_converter_router, prefix="", tags=["document-converter"])
