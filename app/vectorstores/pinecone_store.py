from app.vectorstores.faiss_store import FAISSStore

class PineconeStore(FAISSStore):
    """Fallback Pinecone store running locally via in-memory FAISSStore."""
    def __init__(self):
        super().__init__()
