"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional


class EmbedRequest(BaseModel):
    """Request model for embedding endpoint."""
    text: Optional[str] = Field(
        default=None,
        description="Single text to embed",
        examples=["Hello, world!"]
    )
    texts: Optional[list[str]] = Field(
        default=None,
        description="Multiple texts to embed",
        examples=[["Hello", "World"]]
    )
    
    def get_texts(self) -> list[str]:
        """Get list of texts to embed."""
        if self.texts:
            return self.texts
        if self.text:
            return [self.text]
        return []


class EmbedResponse(BaseModel):
    """Response model for embedding endpoint."""
    embeddings: list[list[float]] = Field(
        description="List of embedding vectors"
    )
    dimension: int = Field(
        description="Dimension of each embedding vector"
    )
    num_texts: int = Field(
        description="Number of texts that were embedded"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether model is loaded")
    model_name: str = Field(description="Name of the loaded model")


class StatsResponse(BaseModel):
    """Response model for service statistics."""
    model_name: str
    device: str
    batch_size: int
    batch_timeout: float
    total_requests: int
    total_batches: int
    total_texts_processed: int
    queue_size: int
    cache: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")
