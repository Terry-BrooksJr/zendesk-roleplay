"""
Caching layer for sentence transformer embeddings.

Provides LRU cache and disk persistence for computed embeddings
to improve intent classification performance.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from loguru import logger


class EmbeddingCache:
    """LRU cache with disk persistence for sentence transformer embeddings."""

    def __init__(
        self, cache_dir: str = ".cache/embeddings", max_memory_size: int = 1000
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_size = max_memory_size
        self._memory_cache: Dict[str, torch.Tensor] = {}
        self._access_order = []  # For LRU tracking

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate a cache key for the text and model combination."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_disk_path(self, cache_key: str) -> Path:
        """Get the disk cache file path for a given cache key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_from_disk(self, cache_key: str) -> Optional[torch.Tensor]:
        """Load embedding from disk cache."""
        disk_path = self._get_disk_path(cache_key)
        if disk_path.exists():
            try:
                with open(disk_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding from disk: {e}")
                # Remove corrupted cache file
                try:
                    disk_path.unlink()
                except Exception:
                    pass
        return None

    def _save_to_disk(self, cache_key: str, embedding: torch.Tensor):
        """Save embedding to disk cache."""
        disk_path = self._get_disk_path(cache_key)
        try:
            with open(disk_path, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding to disk: {e}")

    def _update_access_order(self, cache_key: str):
        """Update LRU access order."""
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
        self._access_order.append(cache_key)

        # Evict oldest if over limit
        while len(self._access_order) > self.max_memory_size:
            oldest_key = self._access_order.pop(0)
            self._memory_cache.pop(oldest_key, None)

    def get(self, text: str, model_name: str) -> Optional[torch.Tensor]:
        """Retrieve embedding from cache (memory first, then disk)."""
        cache_key = self._get_cache_key(text, model_name)

        # Check memory cache first
        if cache_key in self._memory_cache:
            self._update_access_order(cache_key)
            return self._memory_cache[cache_key]

        # Check disk cache
        embedding = self._load_from_disk(cache_key)
        if embedding is not None:
            # Add to memory cache
            self._memory_cache[cache_key] = embedding
            self._update_access_order(cache_key)
            return embedding

        return None

    def put(self, text: str, model_name: str, embedding: torch.Tensor):
        """Store embedding in cache (both memory and disk)."""
        cache_key = self._get_cache_key(text, model_name)

        # Store in memory
        self._memory_cache[cache_key] = embedding
        self._update_access_order(cache_key)

        # Store on disk
        self._save_to_disk(cache_key, embedding)

    def clear(self):
        """Clear all cached embeddings."""
        self._memory_cache.clear()
        self._access_order.clear()

        # Clear disk cache
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")

    def cache_stats(self) -> Dict[str, Union[int, str]]:
        """Get cache statistics."""
        disk_files = len(list(self.cache_dir.glob("*.pkl")))
        cache_size_mb = sum(
            file.stat().st_size for file in self.cache_dir.glob("*.pkl")
        ) / (1024 * 1024)

        return {
            "memory_cache_size": len(self._memory_cache),
            "disk_cache_files": disk_files,
            "cache_size_mb": round(cache_size_mb, 2),
            "cache_directory": str(self.cache_dir),
        }


# Global cache instance
_embedding_cache = EmbeddingCache()


def get_cached_embedding(text: str, model_name: str) -> Optional[torch.Tensor]:
    """Get embedding from global cache."""
    return _embedding_cache.get(text, model_name)


def cache_embedding(text: str, model_name: str, embedding: torch.Tensor):
    """Store embedding in global cache."""
    _embedding_cache.put(text, model_name, embedding)


def clear_embedding_cache():
    """Clear the global embedding cache."""
    _embedding_cache.clear()


def get_cache_stats() -> Dict[str, Union[int, str]]:
    """Get global cache statistics."""
    return _embedding_cache.cache_stats()
