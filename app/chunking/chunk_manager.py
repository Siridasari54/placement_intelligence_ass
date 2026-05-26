from typing import List, Dict, Any
from app.chunking.recursive_chunker import RecursiveChunker
from app.utils.logger import logger

class ChunkManager:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes document inputs and returns standard text chunk dictionary structures."""
        chunks = []
        for doc in docs:
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            
            splits = self.chunker.split_text(text)
            for split in splits:
                chunks.append({
                    "text": split["text"],
                    "metadata": {**meta, **split.get("metadata", {})}
                })
        return chunks

    def chunk_all_data(self) -> List[Dict[str, Any]]:
        """
        Fits backward compatibility requirements in IngestionService.
        Chunks official static JSON data tables from processed directory to index them in the database!
        """
        logger.info("ChunkManager chunking all structured dataset tables...")
        import json
        import os
        
        chunks = []
        processed_dir = "data/processed"
        
        # 1. Chunk Eligibility Profiles
        elig_path = os.path.join(processed_dir, "eligibility_data.json")
        if os.path.exists(elig_path):
            with open(elig_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for row in data:
                comp = row["company"]
                text = (
                    f"Official Placement Profile for {comp}:\n"
                    f"- Minimum CGPA Cutoff is {row['min_cgpa']}.\n"
                    f"- Max Backlogs Allowed is {row['max_backlogs']}.\n"
                    f"- Package offered is {row['package_lpa']} LPA.\n"
                    f"- Service Bond is {row['bond_years']} years.\n"
                    f"- Primary Tech Focus: {row['tech_focus']}.\n"
                    f"- Key Topics to prepare: {row['key_topics']}."
                )
                chunks.append({
                    "text": text,
                    "metadata": {
                        "company": comp,
                        "section": "eligibility",
                        "source": "eligibility_data.json",
                        "min_cgpa": row["min_cgpa"],
                        "package_lpa": row["package_lpa"],
                        "max_backlogs": row["max_backlogs"],
                        "bond_years": row["bond_years"]
                    }
                })
                
        # 2. Chunk Conflict Records
        conflict_path = os.path.join(processed_dir, "conflict_data.json")
        if os.path.exists(conflict_path):
            with open(conflict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            from app.chunking.conflict_chunker import ConflictChunker
            for row in data:
                chunk = ConflictChunker.chunk_conflict_row(row)
                chunks.append(chunk)
                
        # 3. Chunk Placement Trends
        trend_path = os.path.join(processed_dir, "trend_data.json")
        if os.path.exists(trend_path):
            with open(trend_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            from app.chunking.temporal_chunker import TemporalChunker
            for row in data:
                chunks.extend(TemporalChunker.chunk_trend_row(row))
                
        logger.info(f"Chunked structured data successfully. Generated {len(chunks)} DB chunks.")
        return chunks