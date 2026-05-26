from functools import lru_cache


class LRUCache:

    @staticmethod
    @lru_cache(maxsize=128)
    def get_cached_response(
        query: str
    ):

        return None