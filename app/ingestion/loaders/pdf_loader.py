import os
from typing import Dict, Any, List
import pdfplumber
from pypdf import PdfReader
from app.ingestion.base_loader import BaseLoader
from app.utils.logger import logger

class PDFLoader(BaseLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def load(self) -> Dict[str, Any]:
        """
        Loads PDF document, extracting raw text and tabular structures page-by-page.
        """
        logger.info(f"PDFLoader parsing: {self.file_path}")
        
        pages_data = []
        all_tables = []
        combined_text_list = []
        
        # 1. First, parse with pypdf for clean general text extraction
        try:
            reader = PdfReader(self.file_path)
            pypdf_pages = {i + 1: page.extract_text() for i, page in enumerate(reader.pages)}
        except Exception as e:
            logger.warning(f"pypdf reader failed, falling back: {e}")
            pypdf_pages = {}
            
        # 2. Parse with pdfplumber to extract tables and page contents
        try:
            with pdfplumber.open(self.file_path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    page_num = idx + 1
                    
                    # Text extraction
                    text = page.extract_text() or pypdf_pages.get(page_num, "")
                    
                    # Table extraction
                    tables = page.extract_tables()
                    page_tables = []
                    for t in tables:
                        # Clean and serialize table
                        cleaned_table = []
                        for row in t:
                            cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                            cleaned_table.append(cleaned_row)
                        if cleaned_table:
                            page_tables.append(cleaned_table)
                            all_tables.append(cleaned_table)
                            
                    pages_data.append({
                        "page_num": page_num,
                        "text": text,
                        "tables": page_tables
                    })
                    combined_text_list.append(text)
        except Exception as e:
            logger.error(f"pdfplumber failed: {e}")
            # Minimal fallback using pypdf reader if pdfplumber fails
            if pypdf_pages:
                for page_num, text in pypdf_pages.items():
                    pages_data.append({
                        "page_num": page_num,
                        "text": text,
                        "tables": []
                    })
                    combined_text_list.append(text)
                    
        combined_text = "\n\n".join(combined_text_list)
        
        return {
            "text": combined_text,
            "pages": pages_data,
            "tables": all_tables,
            "metadata": {
                "source": os.path.basename(self.file_path),
                "path": self.file_path,
                "type": "pdf",
                "total_pages": len(pages_data)
            }
        }
