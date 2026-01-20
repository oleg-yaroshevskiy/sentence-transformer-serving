"""Caching layer for embeddings with TTL support."""

import time
import hashlib
import threading
from collections import OrderedDict
from typing import Optional
import numpy as np

from config import get_settings


class EmbeddingCache:
    """LRU cache with TTL for embeddings.
    
    Uses an OrderedDict for O(1) access and LRU eviction.
    Thread-safe with a read-write lock pattern.
    """
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Create a hash key for the text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if exists and not expired."""
        key = self._hash_text(text)
        
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            embedding, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return embedding.copy()
    
    def get_many(self, texts: list[str]) -> dict[str, Optional[np.ndarray]]:
        """Get multiple embeddings from cache."""
        results = {}
        for text in texts:
            results[text] = self.get(text)
        return results
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = self._hash_text(text)
        
        with self._lock:
            # If key exists, update it
            if key in self._cache:
                self._cache[key] = (embedding.copy(), time.time())
                self._cache.move_to_end(key)
                return
            
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = (embedding.copy(), time.time())
    
    def set_many(self, texts: list[str], embeddings: np.ndarray) -> None:
        """Store multiple embeddings in cache."""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            }


# Global cache instance
_cache_instance: Optional[EmbeddingCache] = None


def get_cache() -> EmbeddingCache:
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        settings = get_settings()
        _cache_instance = EmbeddingCache(
            max_size=settings.cache_max_size,
            ttl=settings.cache_ttl,
        )
    return _cache_instance
