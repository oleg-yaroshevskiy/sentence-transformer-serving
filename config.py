"""Configuration management for the embedding service."""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model settings
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name"
    )
    
    # Batching settings
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Maximum batch size for processing"
    )
    batch_timeout: float = Field(
        default=0.1,
        ge=0.01,
        le=5.0,
        description="Maximum time to wait for batch to fill (seconds)"
    )
    
    # Cache settings
    cache_enabled: bool = Field(
        default=True,
        description="Enable/disable caching"
    )
    cache_max_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum number of cached embeddings"
    )
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="Cache TTL in seconds"
    )
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=16, description="Number of workers")
    
    # Device settings
    device: str = Field(
        default="cpu",
        description="Device to run model on (cpu/cuda/mps)"
    )
    
    model_config = {
        "env_prefix": "EMBEDDING_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
