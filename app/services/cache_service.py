import sqlite3
from app.cache.cache_manager import cache_manager

class CacheService:
    @staticmethod
    def get_stats() -> dict:
        """Fetches total rows from cache table."""
        cache = cache_manager.get_cache()
        if not cache:
            return {"enabled": False, "count": 0}
            
        try:
            conn = sqlite3.connect(cache.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM cache")
            count = cursor.fetchone()[0]
            conn.close()
            return {"enabled": True, "count": count, "db_path": cache.db_path}
        except Exception:
            return {"enabled": True, "count": 0, "db_path": getattr(cache, "db_path", "unknown")}
