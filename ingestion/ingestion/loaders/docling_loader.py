import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Any
from app.utils.logger import logger
from app.ingestion.base_loader import BaseLoader

class DoclingLoader(BaseLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def load(self) -> Dict[str, Any]:
        """Loads document using Docling, with robust pure-Python fallbacks to avoid import crashes."""
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(self.file_path)
            
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Docling parsed {self.file_path} successfully.")
            return {
                "text": markdown_content,
                "pages": [{"page_num": 1, "text": markdown_content, "tables": []}],
                "tables": [],
                "metadata": {
                    "source": os.path.basename(self.file_path),
                    "path": self.file_path,
                    "parser": "docling"
                }
            }
        except Exception as e:
            logger.warning(f"Docling not available or failed: {e}. Running custom fallback parser.")
            
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext == ".pdf":
                from app.ingestion.loaders.pdf_loader import PDFLoader
                loader = PDFLoader(self.file_path)
                return loader.load()
                
            elif ext == ".docx":
                # Pure-Python docx extraction using zipfile and xml (zero dependencies!)
                extracted_text = self._read_docx_pure_python()
                return {
                    "text": extracted_text,
                    "pages": [{"page_num": 1, "text": extracted_text, "tables": []}],
                    "tables": [],
                    "metadata": {
                        "source": os.path.basename(self.file_path),
                        "path": self.file_path,
                        "parser": "pure-python-docx-fallback"
                    }
                }
            else:
                # Text loader fallback
                try:
                    with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    return {
                        "text": content,
                        "pages": [{"page_num": 1, "text": content, "tables": []}],
                        "tables": [],
                        "metadata": {
                            "source": os.path.basename(self.file_path),
                            "path": self.file_path,
                            "parser": "text-fallback"
                        }
                    }
                except Exception as read_err:
                    logger.error(f"Fallback text loader failed: {read_err}")
                    raise read_err

    def _read_docx_pure_python(self) -> str:
        """Parses docx layout structure without python-docx library using built-in zipfile/xml parser."""
        try:
            with zipfile.ZipFile(self.file_path) as docx:
                xml_content = docx.read('word/document.xml')
                root = ET.fromstring(xml_content)
                
                # Word processing namespaces
                namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                
                paragraphs = []
                # Traverse XML elements looking for paragraphs (p) and run text tags (t)
                for paragraph in root.iter(namespace + 'p'):
                    p_text = []
                    for run in paragraph.iter(namespace + 't'):
                        if run.text:
                            p_text.append(run.text)
                    if p_text:
                        paragraphs.append("".join(p_text))
                        
                return "\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Pure-Python docx parser failed: {e}")
            return ""
