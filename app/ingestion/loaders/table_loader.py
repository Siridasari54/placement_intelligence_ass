import os
import pandas as pd
from typing import Dict, Any, List
from app.utils.logger import logger

class TableLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Table file not found at: {file_path}")

    def load(self) -> List[Dict[str, Any]]:
        """Loads CSV/XLSX file and returns rows as dictionaries alongside markdown text."""
        try:
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext == ".csv":
                df = pd.read_csv(self.file_path)
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(self.file_path)
            else:
                raise ValueError(f"Unsupported table format: {ext}")
            
            rows = df.to_dict(orient="records")
            markdown_table = df.to_markdown(index=False)
            
            logger.info(f"Successfully loaded table: {self.file_path} ({len(rows)} rows)")
            return [{
                "text": markdown_table,
                "rows": rows,
                "metadata": {
                    "source": self.file_path,
                    "rows_count": len(rows),
                    "type": "table"
                }
            }]
        except Exception as e:
            logger.error(f"Error loading table {self.file_path}: {e}")
            raise e
