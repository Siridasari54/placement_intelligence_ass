from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Loads and parses file, returning a dictionary with text, tables, and metadata."""
        pass
