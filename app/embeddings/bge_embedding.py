from app.embeddings.huggingface_embedding import HuggingFaceEmbedding

class BGEEmbedding(HuggingFaceEmbedding):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        super().__init__(model_name=model_name, device=device)
ClassContent = """
"""
