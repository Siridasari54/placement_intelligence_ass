import os
import json
from typing import List, Dict, Any
from app.utils.logger import logger

class ConflictRetriever:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir

    def retrieve_conflict_info(self, company: str) -> List[Dict[str, Any]]:
        """Retrieves official and portal scraps for a target company to highlight conflicts."""
        context_pieces = []
        
        # Load official eligibility data
        elig_file = os.path.join(self.processed_dir, "eligibility_data.json")
        conflict_file = os.path.join(self.processed_dir, "conflict_data.json")
        
        official_record = None
        portal_record = None
        
        if os.path.exists(elig_file):
            with open(elig_file, "r") as f:
                elig_data = json.load(f)
            for row in elig_data:
                if row["company"].lower() == company.lower():
                    official_record = row
                    break
                    
        if os.path.exists(conflict_file):
            with open(conflict_file, "r") as f:
                conf_data = json.load(f)
            for row in conf_data:
                if row["company"].lower() == company.lower():
                    portal_record = row
                    break
                    
        if official_record and portal_record:
            text = (
                f"Conflict Detection Alert! Multiple sources exist for {company}:\n"
                f"- Official Placement Profile: Minimum CGPA Cutoff is {official_record['min_cgpa']}, "
                f"Package is {official_record['package_lpa']} LPA.\n"
                f"- Scraped Unofficial Placement Portal: Cutoff is {portal_record['cgpa_portal']} CGPA, "
                f"Package is {portal_record['package_portal']} LPA.\n"
                f"Conflict Type: {portal_record['conflict_type']}.\n"
                f"IMPORTANT: Prompt the user about this conflict and advise verification."
            )
            context_pieces.append({
                "text": text,
                "metadata": {
                    "company": company,
                    "section": "conflict",
                    "source": "multi-source-detector",
                    "conflict": True
                }
            })
            logger.info(f"Detected conflict data for {company} and compiled conflict chunk.")
            
        elif official_record:
            text = (
                f"Official Profile for {company}: Minimum CGPA Cutoff is {official_record['min_cgpa']}, "
                f"Package is {official_record['package_lpa']} LPA."
            )
            context_pieces.append({
                "text": text,
                "metadata": {"company": company, "section": "eligibility", "source": "official"}
            })
            
        return context_pieces
ClassContent = """
"""
