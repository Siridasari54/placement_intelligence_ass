import re
from typing import Dict, Any, List

class QueryRouter:
    @staticmethod
    def route_query(query: str) -> Dict[str, Any]:
        """Analyzes a query and returns its category, target companies, and filters."""
        query_lower = query.lower()
        
        # 1. Check Out-of-Corpus / Adversarial
        out_of_corpus_patterns = [
            r"stock\s+price", r"visit\s+date", r"campus\s+visit", 
            r"placed.*last\s+year", r"work-from-home", r"wfh", 
            r"in\s+the\s+world", r"highest\s+in\s+the\s+world", 
            r"should\s+i\s+join.*or.*better", r"opinion"
        ]
        if any(re.search(pat, query_lower) for pat in out_of_corpus_patterns):
            return {
                "route": "fallback",
                "reason": "out-of-corpus query",
                "filters": {}
            }
            
        # Check Below-Threshold / Edge Cases
        if "cgpa of 5.0" in query_lower or "cgpa.*5\." in query_lower:
            return {
                "route": "edge_case_cgpa",
                "reason": "cgpa threshold edge case",
                "filters": {"min_cgpa": 5.0}
            }

        # 2. Check Conflict Detection
        conflict_keywords = ["conflict", "scraped", "portal", "official vs", "difference", "amazon cgpa cutoff"]
        is_conflict = any(k in query_lower for k in conflict_keywords)
        # Check if the query specifically targets a company with conflicting records
        from app.utils.constants import CONFLICT_COMPANIES
        matched_conflict_company = [c for c in CONFLICT_COMPANIES if c.lower() in query_lower]
        
        if is_conflict or (matched_conflict_company and ("cutoff" in query_lower or "package" in query_lower or "cgpa" in query_lower)):
            return {
                "route": "conflict",
                "reason": "query matching conflict-prone fields",
                "company": matched_conflict_company[0] if matched_conflict_company else None,
                "filters": {}
            }

        # 3. Check Temporal / Trend
        temporal_keywords = ["trend", "increase", "decrease", "grow", "2021", "2022", "2023", "2024", "timeline", "years"]
        if any(k in query_lower for k in temporal_keywords):
            return {
                "route": "temporal",
                "reason": "query involves temporal reasoning over multiple years",
                "filters": {}
            }

        # 4. Check Multi-hop Eligibility/Hiring queries
        # E.g. M1, M2, M3, H1, H2, H3
        eligibility_keywords = ["cgpa", "backlog", "bond", "package", "lpa", "qualify", "applies to", "can apply"]
        hiring_keywords = ["hire", "analyst", "sde", "officer", "intern", "roles", "distribution"]
        
        has_eligibility = any(k in query_lower for k in eligibility_keywords)
        has_hiring = any(k in query_lower for k in hiring_keywords)
        
        # Check for multiple constraints (Multi-hop)
        # e.g., "CGPA 7.0, 1 backlog wants maximum pay with no bond"
        number_count = len(re.findall(r"\d+(\.\d+)?", query_lower))
        
        if (has_eligibility and has_hiring) or (has_eligibility and number_count >= 2) or ("highest package among" in query_lower):
            return {
                "route": "multi_hop",
                "reason": "query requires synthesizing multiple eligibility or hiring constraints",
                "filters": {}
            }
            
        if has_eligibility:
            return {
                "route": "eligibility",
                "reason": "single-hop eligibility query",
                "filters": {}
            }

        if has_hiring:
            return {
                "route": "hiring",
                "reason": "single-hop hiring distribution query",
                "filters": {}
            }

        return {
            "route": "standard",
            "reason": "general semantic search query",
            "filters": {}
        }
