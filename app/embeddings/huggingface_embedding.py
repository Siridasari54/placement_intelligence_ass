from langchain_huggingface import HuggingFaceEmbeddings


class HuggingFaceEmbedding:

    def load(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )