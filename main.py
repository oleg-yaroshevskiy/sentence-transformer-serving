"""FastAPI application for serving SentenceTransformer embeddings."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings, Settings
from service import get_embedding_service, EmbeddingService
from models import (
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting embedding service...")
    service = get_embedding_service()
    await service.start()
    logger.info("Embedding service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down embedding service...")
    await service.stop()
    logger.info("Embedding service stopped")


# Create FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="High-performance SentenceTransformer embedding service with batching and caching",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> EmbeddingService:
    """Dependency to get the embedding service."""
    return get_embedding_service()


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(service: EmbeddingService = Depends(get_service)):
    """Check if the service is healthy and model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=service._model is not None,
        model_name=service.settings.model_name,
    )


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_stats(service: EmbeddingService = Depends(get_service)):
    """Get service statistics including cache hit rates."""
    stats = service.get_stats()
    return StatsResponse(**stats)


@app.post("/embed", response_model=EmbedResponse, tags=["Embeddings"])
async def embed(
    request: EmbedRequest,
    service: EmbeddingService = Depends(get_service)
):
    """Generate embeddings for text(s).
    
    Supports both single text and batch text input:
    - Use `text` field for a single text
    - Use `texts` field for multiple texts (more efficient for batches)
    
    Features:
    - **Automatic batching**: Requests are batched for efficient GPU utilization
    - **Caching**: Repeated texts are served from cache
    - **Normalized embeddings**: All embeddings are L2-normalized
    """
    texts = request.get_texts()
    
    if not texts:
        raise HTTPException(
            status_code=400,
            detail="Either 'text' or 'texts' must be provided"
        )
    
    try:
        if len(texts) == 1:
            embedding = await service.encode(texts[0])
            embeddings = [embedding.tolist()]
        else:
            results = await service.encode_many(texts)
            embeddings = [e.tolist() for e in results]
        
        return EmbedResponse(
            embeddings=embeddings,
            dimension=service.get_embedding_dimension(),
            num_texts=len(texts),
        )
    
    except Exception as e:
        logger.exception("Error generating embeddings")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/embed/single", response_model=EmbedResponse, tags=["Embeddings"])
async def embed_single(
    text: str,
    service: EmbeddingService = Depends(get_service)
):
    """Generate embedding for a single text (query parameter version).
    
    Simple endpoint for single text embedding via query parameter.
    """
    try:
        embedding = await service.encode(text)
        return EmbedResponse(
            embeddings=[embedding.tolist()],
            dimension=service.get_embedding_dimension(),
            num_texts=1,
        )
    except Exception as e:
        logger.exception("Error generating embedding")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.delete("/cache", tags=["Cache"])
async def clear_cache(service: EmbeddingService = Depends(get_service)):
    """Clear the embedding cache."""
    if service.settings.cache_enabled:
        service.cache.clear()
        return {"message": "Cache cleared"}
    return {"message": "Cache is disabled"}


@app.get("/config", tags=["Configuration"])
async def get_config(settings: Settings = Depends(get_settings)):
    """Get current service configuration."""
    return {
        "model_name": settings.model_name,
        "device": settings.device,
        "batch_size": settings.batch_size,
        "batch_timeout": settings.batch_timeout,
        "cache_enabled": settings.cache_enabled,
        "cache_max_size": settings.cache_max_size,
        "cache_ttl": settings.cache_ttl,
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=False,
    )
