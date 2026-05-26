from app.cache.sqlite_cache import SQLitePersistentCache
from app.utils.config_loader import config_loader
from app.utils.logger import logger

class CacheManager:
    _instance = None
    _cache: SQLitePersistentCache = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            threshold = config_loader.get("cache.semantic.threshold", 0.85)
            db_path = config_loader.get("cache.sqlite.db_path", "data/cache/semantic_cache.db")
            cls._instance._cache = SQLitePersistentCache(db_path=db_path, threshold=threshold)
        return cls._instance

    def get_cache(self) -> SQLitePersistentCache:
        """Returns the underlying SQLite Persistent Cache instance."""
        return self._cache

    def get(self, query: str):
        """Standard proxy interface for getting a cached answer."""
        return self._cache.get(query)

    def save(self, query: str, response: str):
        """Standard proxy interface for saving an answer."""
        self._cache.set(query, response)

    def set(self, query: str, response: str):
        """Alias for save/set compatibility."""
        self._cache.set(query, response)

# Global singleton instance
cache_manager = CacheManager()