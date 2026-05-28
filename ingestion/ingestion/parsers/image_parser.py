import os
from typing import Dict, Any
from app.ingestion.parsers.chart_parser import ChartParser

class ImageParser:
    @staticmethod
    def parse_image(file_path: str) -> Dict[str, Any]:
        """Generic image parser. Returns textual descriptors and classification metadata."""
        basename = os.path.basename(file_path)
        is_chart = "chart" in basename.lower() or "distribution" in basename.lower()
        
        caption = ""
        if is_chart:
            caption = ChartParser.parse_chart_to_text(file_path)
            
        return {
            "file_name": basename,
            "is_chart": is_chart,
            "caption": caption,
            "metadata": {
                "source": file_path,
                "type": "image",
                "is_chart": is_chart
            }
        }
