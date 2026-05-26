import os
from typing import List, Dict, Any
from app.utils.logger import logger
from app.ingestion.loaders.multi_loader import MultiLoader
from app.utils.metadata_utils import extract_metadata_from_text

class IngestionPipeline:
    def __init__(self):
        self.loader = MultiLoader()

    def run(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Runs the ingestion pipeline over a list of document file paths.
        Parses pages, extracts tables, chunks the text, and enriches metadata.
        Returns a list of structured chunk dicts.
        """
        logger.info(f"IngestionPipeline processing {len(file_paths)} files...")
        
        all_chunks = []
        
        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}, skipping.")
                continue
                
            try:
                # 1. Load raw file
                doc_data = self.loader.load_file(path)
                
                # 2. Extract metadata from general document
                filename = os.path.basename(path)
                logger.info(f"Loaded: '{filename}' successfully. Splitting into chunks...")
                
                # 3. Create chunks page by page (retaining page numbers!)
                pages = doc_data.get("pages", [])
                
                from app.chunking.recursive_chunker import RecursiveChunker
                chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
                
                for page in pages:
                    page_num = page.get("page_num", 1)
                    page_text = page.get("text", "").strip()
                    
                    if not page_text:
                        continue
                        
                    # Split page text into sub-chunks
                    sub_chunks = chunker.split_text(page_text)
                    
                    for sub_c in sub_chunks:
                        chunk_text = sub_c["text"]
                        
                        # Enrich metadata based on content
                        meta = extract_metadata_from_text(chunk_text)
                        
                        # Apply standard files source, page and type details
                        meta["source"] = filename
                        meta["page"] = page_num
                        meta["type"] = doc_data["metadata"].get("type", "pdf")
                        
                        all_chunks.append({
                            "text": chunk_text,
                            "metadata": meta
                        })
                        
                # 4. Process tables as separate chunks
                tables = doc_data.get("tables", [])
                for t_idx, table in enumerate(tables):
                    # Convert table grid into structured markdown description
                    # Table loader might already have it, let's serialize it safely
                    table_str = "Structured Tabular Data:\n"
                    for row in table:
                        if isinstance(row, dict):
                            table_str += " | ".join([f"{k}: {v}" for k, v in row.items()]) + "\n"
                        elif isinstance(row, list):
                            table_str += " | ".join([str(cell) for cell in row]) + "\n"
                            
                    meta = extract_metadata_from_text(table_str)
                    meta["source"] = filename
                    meta["page"] = 1  # Tables defaulted to page 1
                    meta["type"] = "table"
                    
                    all_chunks.append({
                        "text": table_str,
                        "metadata": meta
                    })
                    
            except Exception as e:
                logger.error(f"Error processing file '{path}' in IngestionPipeline: {e}")
                
        logger.info(f"Ingestion pipeline generated {len(all_chunks)} chunks.")
        return all_chunks

    def process_raw_dataset_text(self, raw_text: str) -> Dict[str, Any]:
        """Backward compatibility helper for parsing raw placement dataset text."""
        # Simple rule-based mock parser for testing
        logger.info("process_raw_dataset_text parsing raw input...")
        return {
            "eligibility": [{"company": "Google", "min_cgpa": 7.4, "package_lpa": 42.0}]
        }