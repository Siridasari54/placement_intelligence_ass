class RedisCache:

    def __init__(self):

        self.cache = {}

    def set(
        self,
        key,
        value
    ):

        self.cache[key] = value

    def get(
        self,
        key
    ):

        return self.cache.get(key)

    def exists(
        self,
        key
    ):

        return key in self.cache