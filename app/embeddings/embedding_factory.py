from app.embeddings.huggingface_embedding import HuggingFaceEmbedding

class EmbeddingFactory:
    _instances = {}

    @classmethod
    def get_embedding(cls, model_name="huggingface"):
        if model_name not in cls._instances:
            if model_name == "huggingface":
                cls._instances[model_name] = HuggingFaceEmbedding().load()
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")
        return cls._instances[model_name]

# Standard shared embedding instance used globally
embedding_model = EmbeddingFactory.get_embedding("huggingface")