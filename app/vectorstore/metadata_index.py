from typing import List, Set, Dict, Any

class MetadataIndex:
    """Manages unique tags and attributes from ingested data to facilitate quick metadata filtering."""
    def __init__(self):
        self.companies: Set[str] = set()
        self.sections: Set[str] = set()
        self.years: Set[int] = set()

    def update(self, chunks: List[Dict[str, Any]]) -> None:
        """Updates the internal index based on new document chunks."""
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            if "company" in meta:
                self.companies.add(meta["company"])
            if "section" in meta:
                self.sections.add(meta["section"])
            if "year" in meta:
                try:
                    self.years.add(int(meta["year"]))
                except (ValueError, TypeError):
                    pass

    def get_available_companies(self) -> List[str]:
        return sorted(list(self.companies))

    def get_available_sections(self) -> List[str]:
        return sorted(list(self.sections))

    def get_available_years(self) -> List[int]:
        return sorted(list(self.years))
