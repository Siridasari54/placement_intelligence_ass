import os
import json
from typing import List, Dict, Any, Optional
from app.utils.logger import logger


class ConflictRetriever:
    """Retrieves and compares official and portal-scraped data to detect conflicts."""
    
    def __init__(self, processed_dir: str = "data/processed"):
        """Initialize the conflict retriever.
        
        Args:
            processed_dir: Directory containing processed data files
        """
        self.processed_dir = processed_dir

    def _load_json_file(self, filename: str) -> Optional[List[Dict[str, Any]]]:
        """Load JSON data from file with error handling.
        
        Args:
            filename: Name of the JSON file to load
            
        Returns:
            List of dictionaries if successful, None otherwise
        """
        filepath = os.path.join(self.processed_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None

    def _find_company_record(self, data: List[Dict[str, Any]], company: str) -> Optional[Dict[str, Any]]:
        """Find a company record in data list.
        
        Args:
            data: List of company records
            company: Company name to search for
            
        Returns:
            Company record if found, None otherwise
        """
        if not data:
            return None
        
        company_lower = company.lower()
        for row in data:
            if row.get("company", "").lower() == company_lower:
                return row
        return None

    def retrieve_conflict_info(self, company: str) -> List[Dict[str, Any]]:
        """Retrieves official and portal scraps for a target company to highlight conflicts.
        
        Args:
            company: Company name to check for conflicts
            
        Returns:
            List of context pieces with conflict information
        """
        context_pieces = []
        
        # Load official eligibility data
        elig_data = self._load_json_file("eligibility_data.json")
        conf_data = self._load_json_file("conflict_data.json")
        
        if not elig_data:
            logger.warning("Could not load eligibility data for conflict detection")
            return context_pieces
        
        # Find records
        official_record = self._find_company_record(elig_data, company)
        portal_record = self._find_company_record(conf_data, company) if conf_data else None
        
        # Generate conflict chunk if both sources exist
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
        
        # Return official data if no conflict found
        elif official_record:
            text = (
                f"Official Profile for {company}: Minimum CGPA Cutoff is {official_record['min_cgpa']}, "
                f"Package is {official_record['package_lpa']} LPA."
            )
            context_pieces.append({
                "text": text,
                "metadata": {
                    "company": company,
                    "section": "eligibility",
                    "source": "official"
                }
            })
        
        return context_pieces
