from app.vectorstores.faiss_store import FAISSStore

class QdrantStore(FAISSStore):
    """Fallback Qdrant store running locally via in-memory FAISSStore."""
    def __init__(self):
        super().__init__()
