from app.llm.groq_llm import GroqLLM
from app.utils.config_loader import config_loader
from app.utils.logger import logger

class LLMFactory:
    _instance = None

    @classmethod
    def get_llm(cls, provider: str = None):
        if provider is None:
            provider = config_loader.get("model.llm.provider", "groq")
        
        provider = provider.lower()
        if provider == "groq":
            return GroqLLM()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
