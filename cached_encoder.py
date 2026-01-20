"""Cached SentenceTransformer wrapper with statistics tracking."""

import logging
import time
import hashlib
import threading
from typing import List, Optional, Dict
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EncoderStats:
    """Statistics for the encoder."""
    total_calls: int = 0
    total_texts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    texts_encoded: int = 0  # actual model calls, not from cache
    
    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_texts": self.total_texts,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "texts_encoded": self.texts_encoded,
        }


class CachedSentenceTransformer:
    """Drop-in replacement for SentenceTransformer with LRU caching.
    
    Has the same interface as SentenceTransformer but adds:
    - LRU cache with configurable max size and TTL
    - Thread-safe operations
    - Statistics tracking
    - Efficient batch processing (only encodes uncached texts)
    
    Usage:
        # Same interface as SentenceTransformer
        encoder = CachedSentenceTransformer("thenlper/gte-large")
        
        # Single text
        embedding = encoder.encode("Hello world")
        
        # Multiple texts (efficient - only encodes uncached)
        embeddings = encoder.encode(["Hello", "World", "Hello"])
        
        # Check cache stats
        print(encoder.get_cache_stats())
    """

    def __init__(
        self,
        model_name_or_path: str = "thenlper/gte-large",
        device: Optional[str] = None,
        cache_max_size: int = 10000,
        cache_ttl: Optional[int] = None,  # None = no expiration
        **kwargs,
    ):
        """Initialize the cached encoder.
        
        Args:
            model_name_or_path: SentenceTransformer model name or path
            device: Device to run on (cpu/cuda/mps), None for auto
            cache_max_size: Maximum number of cached embeddings
            cache_ttl: Cache TTL in seconds (None for no expiration)
            **kwargs: Additional arguments passed to SentenceTransformer
        """
        self._model_name = model_name_or_path
        self._device = device
        self._model_kwargs = kwargs
        self.cache_max_size = cache_max_size
        self.cache_ttl = cache_ttl
        
        self._model: Optional[SentenceTransformer] = None
        self._cache: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = EncoderStats()
    
    def _load_model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading model: {self._model_name}")
            self._model = SentenceTransformer(
                self._model_name, 
                device=self._device,
                **self._model_kwargs
            )
            logger.info(f"Model loaded on device: {self._model.device}")
        return self._model
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Create a hash key for the text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
    
    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if exists and not expired."""
        key = self._hash_text(text)
        
        with self._lock:
            if key not in self._cache:
                return None
            
            embedding, timestamp = self._cache[key]
            
            # Check TTL if set
            if self.cache_ttl is not None:
                if time.time() - timestamp > self.cache_ttl:
                    del self._cache[key]
                    return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return embedding.copy()
    
    def _set_in_cache(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = self._hash_text(text)
        
        with self._lock:
            if key in self._cache:
                self._cache[key] = (embedding.copy(), time.time())
                self._cache.move_to_end(key)
                return
            
            # Evict oldest if at capacity
            while len(self._cache) >= self.cache_max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = (embedding.copy(), time.time())
    
    def encode(
        self,
        sentences: str | List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences to embeddings (same interface as SentenceTransformer).
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            show_progress_bar: Show progress bar during encoding
            convert_to_numpy: Convert output to numpy array
            convert_to_tensor: Convert output to torch tensor
            normalize_embeddings: Normalize embeddings to unit length
            **kwargs: Additional arguments passed to model.encode()
            
        Returns:
            Embeddings as numpy array or tensor
        """
        # Handle single string
        single_input = isinstance(sentences, str)
        if single_input:
            sentences = [sentences]
        
        with self._lock:
            self._stats.total_calls += 1
            self._stats.total_texts += len(sentences)
        
        # Check cache for each text
        results: Dict[int, np.ndarray] = {}
        texts_to_encode: List[tuple[int, str]] = []
        
        for i, text in enumerate(sentences):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[i] = cached
                with self._lock:
                    self._stats.cache_hits += 1
            else:
                texts_to_encode.append((i, text))
                with self._lock:
                    self._stats.cache_misses += 1
        
        # Encode only uncached texts
        if texts_to_encode:
            indices, uncached_texts = zip(*texts_to_encode)
            
            model = self._load_model()
            embeddings = model.encode(
                list(uncached_texts),
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,  # Always numpy for cache
                convert_to_tensor=False,
                normalize_embeddings=normalize_embeddings,
                **kwargs,
            )
            
            with self._lock:
                self._stats.texts_encoded += len(uncached_texts)
            
            # Store results and cache
            for idx, text, embedding in zip(indices, uncached_texts, embeddings):
                results[idx] = embedding
                self._set_in_cache(text, embedding)
        
        # Reconstruct in original order
        output = np.array([results[i] for i in range(len(sentences))])
        
        # Handle single input
        if single_input:
            output = output[0]
        
        # Convert output format if needed
        if convert_to_tensor:
            import torch
            output = torch.from_numpy(output)
        
        return output
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._load_model().get_sentence_embedding_dimension()
    
    @property
    def device(self):
        """Get the device the model is on."""
        return self._load_model().device
    
    # Cache management methods
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            stats = self._stats.to_dict()
            stats["cache_size"] = len(self._cache)
            stats["cache_max_size"] = self.cache_max_size
            return stats
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        with self._lock:
            self._cache.clear()
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._stats = EncoderStats()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the encoder - same interface as SentenceTransformer
    encoder = CachedSentenceTransformer(
        "all-MiniLM-L6-v2",  # same positional arg as SentenceTransformer
        cache_max_size=1000,
    )
    
    print("=== Single text ===")
    emb1 = encoder.encode("Hello world")
    print(f"Shape: {emb1.shape}")
    print(f"Stats: {encoder.get_cache_stats()}")
    
    print("\n=== Same text again (should be cached) ===")
    emb2 = encoder.encode("Hello world")
    print(f"Match: {np.allclose(emb1, emb2)}")
    print(f"Stats: {encoder.get_cache_stats()}")
    
    print("\n=== Batch with mixed cache ===")
    texts = ["Hello world", "New text", "Another new one", "Hello world"]
    embeddings = encoder.encode(texts)
    print(f"Shape: {embeddings.shape}")
    print(f"Stats: {encoder.get_cache_stats()}")
    
    print("\n=== Final stats ===")
    stats = encoder.get_cache_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
