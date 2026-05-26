import os
from typing import Dict, Any, List
from app.utils.logger import logger
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.docling_loader import DoclingLoader
from app.ingestion.loaders.table_loader import TableLoader

class MultiLoader:
    def __init__(self):
        pass

    def load_file(self, file_path: str) -> Dict[str, Any]:
        """
        Routes file_path to its corresponding loader and returns normalized output structure:
        {
            "text": str,
            "pages": List[Dict[str, Any]],
            "tables": List[List[List[str]]],
            "metadata": Dict[str, Any]
        }
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        logger.info(f"Routing file: {file_path} (extension: {ext})")

        # 1. Excel/CSV Table Loader
        if ext in [".csv", ".xlsx"]:
            loader = TableLoader(file_path)
            results = loader.load()
            if results:
                # TableLoader outputs a list of sheet dicts
                combined_text = "\n\n".join([res["text"] for res in results])
                all_tables = [res["rows"] for res in results]
                return {
                    "text": combined_text,
                    "pages": [{"page_num": 1, "text": res["text"], "tables": [res["rows"]]} for res in results],
                    "tables": all_tables,
                    "metadata": results[0]["metadata"]
                }
            return {"text": "", "pages": [], "tables": [], "metadata": {"source": os.path.basename(file_path)}}

        # 2. PDF Loader
        elif ext == ".pdf":
            # Prefer standard pdfplumber extraction to preserve pages and tables
            loader = PDFLoader(file_path)
            return loader.load()

        # 3. Word/DOCX or Fallback (Docling or plain text)
        elif ext in [".docx", ".txt", ".md"]:
            try:
                # Try docling first if installed and docling loader succeeds
                loader = DoclingLoader(file_path)
                return loader.load()
            except Exception as e:
                logger.warning(f"Docling conversion failed or docling not installed: {e}. Falling back to plain text loader.")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                return {
                    "text": content,
                    "pages": [{"page_num": 1, "text": content, "tables": []}],
                    "tables": [],
                    "metadata": {
                        "source": os.path.basename(file_path),
                        "path": file_path,
                        "type": "text"
                    }
                }
        else:
            # Attempt to read as raw text
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                return {
                    "text": content,
                    "pages": [{"page_num": 1, "text": content, "tables": []}],
                    "tables": [],
                    "metadata": {"source": os.path.basename(file_path), "path": file_path, "type": "text"}
                }
            except Exception as e:
                raise ValueError(f"Unsupported file type: {ext} (Error: {e})")
