import re
from typing import List, Dict, Any
from app.utils.logger import logger
from app.utils.helpers import extract_numeric

class TableParser:
    @staticmethod
    def parse_eligibility_table(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses the Section 1: Company Eligibility Profiles table."""
        parsed = []
        for row in raw_rows:
            # Standardize keys and clean values
            company = row.get("Company", "").strip()
            if not company:
                continue
            
            # Map column names if they are slightly different
            cgpa = extract_numeric(str(row.get("Min CGPA", row.get("CGPA", 0))))
            backlogs = int(extract_numeric(str(row.get("Max Backlogs", row.get("Backlogs", 0)))))
            package = extract_numeric(str(row.get("Package (LPA)", row.get("Package", 0))))
            bond = int(extract_numeric(str(row.get("Bond (Yrs)", row.get("Bond", 0)))))
            key_topics = row.get("Key Topics", "").strip()
            tech_focus = row.get("Tech Focus", "").strip()
            
            parsed.append({
                "company": company,
                "min_cgpa": cgpa,
                "max_backlogs": backlogs,
                "package_lpa": package,
                "bond_years": bond,
                "key_topics": key_topics,
                "tech_focus": tech_focus,
                "section": "eligibility",
                "source": "official"
            })
        return parsed

    @staticmethod
    def parse_hiring_table(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses the Section 3: Hiring Distribution Data Table."""
        parsed = []
        for row in raw_rows:
            company = row.get("Company", "").strip()
            if not company:
                continue
            parsed.append({
                "company": company,
                "sde": int(extract_numeric(str(row.get("SDE", 0)))),
                "analyst": int(extract_numeric(str(row.get("Analyst", 0)))),
                "officer": int(extract_numeric(str(row.get("Officer", 0)))),
                "intern": int(extract_numeric(str(row.get("Intern", 0)))),
                "total": int(extract_numeric(str(row.get("Total", 0)))),
                "section": "hiring",
                "source": "official"
            })
        return parsed

    @staticmethod
    def parse_trend_table(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses Section 5: Placement Trend Data (2021-2024)."""
        parsed = []
        for row in raw_rows:
            company = row.get("Company", "").strip()
            if not company:
                continue
            
            # Extract packages for different years
            p_2021 = extract_numeric(str(row.get("2021 (LPA)", row.get("2021", 0))))
            p_2022 = extract_numeric(str(row.get("2022 (LPA)", row.get("2022", 0))))
            p_2023 = extract_numeric(str(row.get("2023 (LPA)", row.get("2023", 0))))
            p_2024 = extract_numeric(str(row.get("2024 (LPA)", row.get("2024", 0))))
            trend = row.get("3-Year Trend", "").strip()
            
            parsed.append({
                "company": company,
                "package_2021": p_2021,
                "package_2022": p_2022,
                "package_2023": p_2023,
                "package_2024": p_2024,
                "trend_direction": trend,
                "section": "trend",
                "source": "official"
            })
        return parsed

    @staticmethod
    def parse_conflict_table(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses Section 6: Conflicting Data Table."""
        parsed = []
        for row in raw_rows:
            company = row.get("Company", "").strip()
            if not company:
                continue
            
            # Extraction
            cgpa_official = extract_numeric(str(row.get("CGPA (Official)", 0)))
            cgpa_portal = extract_numeric(str(row.get("CGPA (Portal)", 0)))
            package_official = extract_numeric(str(row.get("Package Official", 0)))
            package_portal = extract_numeric(str(row.get("Package Portal", 0)))
            conflict = row.get("Conflict?", "").strip()
            
            parsed.append({
                "company": company,
                "cgpa_official": cgpa_official,
                "cgpa_portal": cgpa_portal,
                "package_official": package_official,
                "package_portal": package_portal,
                "conflict_type": conflict,
                "section": "conflict",
                "source": "portal",
                "conflict_flag": True
            })
        return parsed

    @staticmethod
    def parse_statistics_table(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses Section 7: Overall Placement Statistics Table."""
        parsed = []
        for row in raw_rows:
            company = row.get("Company", "").strip()
            if not company:
                continue
            
            avg_pkg = extract_numeric(str(row.get("Avg Package", 0)))
            max_offers = int(extract_numeric(str(row.get("Max Offers", 0))))
            min_offers = int(extract_numeric(str(row.get("Min Offers", 0))))
            avg_cgpa = extract_numeric(str(row.get("Avg CGPA Cutoff", 0)))
            bond_free = row.get("Bond-free?", "No").strip().lower() == "yes"
            
            parsed.append({
                "company": company,
                "avg_package": avg_pkg,
                "max_offers": max_offers,
                "min_offers": min_offers,
                "avg_cgpa_cutoff": avg_cgpa,
                "bond_free": bond_free,
                "section": "statistics",
                "source": "official"
            })
        return parsed
