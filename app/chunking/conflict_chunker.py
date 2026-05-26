from typing import Dict, Any

class ConflictChunker:
    @staticmethod
    def chunk_conflict_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Serializes a portal conflict row into a text chunk."""
        company = row["company"]
        text = (
            f"Unofficial Placement Portal Scraped Profile for {company}: "
            f"Scraped CGPA requirement cutoff is {row['cgpa_portal']}. "
            f"Scraped package offered is {row['package_portal']} LPA. "
            f"Note: This is unofficial data that may conflict with official data."
        )
        return {
            "text": text,
            "metadata": {
                "company": company,
                "section": "conflict",
                "source": "portal",
                "conflict": True,
                "cgpa_portal": row["cgpa_portal"],
                "package_portal": row["package_portal"],
                "conflict_type": row["conflict_type"]
            }
        }
