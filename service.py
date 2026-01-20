"""Embedding service with dynamic batching support."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from config import get_settings, Settings
from cache import get_cache, EmbeddingCache

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRequest:
    """A single embedding request in the batch queue."""
    text: str
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)


class EmbeddingService:
    """Service for generating embeddings with batching and caching.
    
    Uses dynamic batching to efficiently process multiple requests:
    - Incoming requests are queued
    - Batch is processed when it reaches max size OR timeout expires
    - Cache is checked first to avoid redundant computation
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._model: Optional[SentenceTransformer] = None
        self._queue: asyncio.Queue[EmbeddingRequest] = asyncio.Queue()
        self._batch_task: Optional[asyncio.Task] = None
        self._cache: Optional[EmbeddingCache] = None
        self._is_running = False
        
        # Metrics
        self._total_requests = 0
        self._total_batches = 0
        self._total_texts_processed = 0
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading model: {self.settings.model_name}")
            self._model = SentenceTransformer(
                self.settings.model_name,
                device=self.settings.device
            )
            logger.info(f"Model loaded on device: {self.settings.device}")
        return self._model
    
    @property
    def cache(self) -> EmbeddingCache:
        """Get the cache instance."""
        if self._cache is None:
            self._cache = get_cache()
        return self._cache
    
    async def start(self) -> None:
        """Start the batching background task."""
        if self._is_running:
            return
        
        self._is_running = True
        # Pre-load model
        _ = self.model
        self._batch_task = asyncio.create_task(self._batch_processor())
        logger.info("Embedding service started")
    
    async def stop(self) -> None:
        """Stop the batching background task."""
        self._is_running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        logger.info("Embedding service stopped")
    
    async def _batch_processor(self) -> None:
        """Background task that processes batches of requests."""
        while self._is_running:
            batch: list[EmbeddingRequest] = []
            deadline = None
            
            try:
                # Wait for first request
                request = await self._queue.get()
                batch.append(request)
                deadline = time.time() + self.settings.batch_timeout
                
                # Collect more requests until batch is full or timeout
                while len(batch) < self.settings.batch_size:
                    timeout = max(0, deadline - time.time())
                    if timeout <= 0:
                        break
                    
                    try:
                        request = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=timeout
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break
                
                # Process the batch
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                # Complete any pending requests with an error
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(
                            RuntimeError("Service shutting down")
                        )
                raise
            except Exception as e:
                logger.exception("Error in batch processor")
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)
    
    async def _process_batch(self, batch: list[EmbeddingRequest]) -> None:
        """Process a batch of embedding requests."""
        if not batch:
            return
        
        self._total_batches += 1
        texts = [req.text for req in batch]
        
        # Check cache for each text
        cached_results: dict[str, Optional[np.ndarray]] = {}
        texts_to_encode: list[str] = []
        
        if self.settings.cache_enabled:
            cached_results = self.cache.get_many(texts)
            texts_to_encode = [t for t in texts if cached_results[t] is None]
        else:
            texts_to_encode = texts
        
        # Encode only uncached texts
        new_embeddings: dict[str, np.ndarray] = {}
        if texts_to_encode:
            # Run model inference in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    texts_to_encode,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )
            
            # Map results and update cache
            for text, embedding in zip(texts_to_encode, embeddings):
                new_embeddings[text] = embedding
                if self.settings.cache_enabled:
                    self.cache.set(text, embedding)
        
        # Send results to all waiting requests
        for req in batch:
            try:
                if req.text in new_embeddings:
                    embedding = new_embeddings[req.text]
                else:
                    embedding = cached_results[req.text]
                
                if not req.future.done():
                    req.future.set_result(embedding)
                self._total_texts_processed += 1
            except Exception as e:
                if not req.future.done():
                    req.future.set_exception(e)
        
        logger.debug(
            f"Processed batch: {len(batch)} requests, "
            f"{len(texts_to_encode)} encoded, "
            f"{len(batch) - len(texts_to_encode)} from cache"
        )
    
    async def encode(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding.
        
        Args:
            text: The text to encode.
            
        Returns:
            The embedding as a numpy array.
        """
        self._total_requests += 1
        
        # Quick cache check before queuing
        if self.settings.cache_enabled:
            cached = self.cache.get(text)
            if cached is not None:
                return cached
        
        # Create request and wait for result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request = EmbeddingRequest(text=text, future=future)
        
        await self._queue.put(request)
        return await future
    
    async def encode_many(self, texts: list[str]) -> list[np.ndarray]:
        """Encode multiple texts into embeddings.
        
        Args:
            texts: List of texts to encode.
            
        Returns:
            List of embeddings as numpy arrays.
        """
        # Submit all at once and gather results
        tasks = [self.encode(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        stats = {
            "model_name": self.settings.model_name,
            "device": self.settings.device,
            "batch_size": self.settings.batch_size,
            "batch_timeout": self.settings.batch_timeout,
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "total_texts_processed": self._total_texts_processed,
            "queue_size": self._queue.qsize(),
        }
        if self.settings.cache_enabled:
            stats["cache"] = self.cache.stats()
        return stats
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        return self.model.get_sentence_embedding_dimension()


# Global service instance
_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = EmbeddingService()
    return _service_instance
