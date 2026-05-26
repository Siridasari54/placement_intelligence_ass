import hashlib
from typing import List, Dict, Any

class DuplicateChecker:
    @staticmethod
    def calculate_hash(text: str) -> str:
        """Returns MD5 hash of standardized text."""
        cleaned = "".join(text.lower().split())
        return hashlib.md5(cleaned.encode("utf-8")).hexdigest()

    @classmethod
    def filter_exact_duplicates(cls, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters exact duplicates based on hash of text field."""
        seen = set()
        unique = []
        for doc in documents:
            doc_hash = cls.calculate_hash(doc.get("text", ""))
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique.append(doc)
        return unique
