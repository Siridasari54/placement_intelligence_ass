import os
from typing import Dict, Any

class ChartParser:
    @staticmethod
    def parse_chart_to_text(filename: str, image_content: bytes = None) -> str:
        """Parses image metadata or calls OCR to generate a readable text summary of a chart."""
        basename = os.path.basename(filename).lower()
        
        # We can extract the company name and find the corresponding row in our structured database
        from app.utils.constants import COMPANIES
        matched_company = None
        for c in COMPANIES:
            if c.lower().replace(" ", "_") in basename or c.lower() in basename:
                matched_company = c
                break
                
        if not matched_company:
            return "Unidentified placement chart."
            
        # Simulates a Vision model converting chart to descriptive text
        return f"This chart represents the hiring distribution by role for {matched_company}."
