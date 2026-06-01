"""Opinion Guard for objective handling of subjective placement queries."""

import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class OpinionGuard:
    """Interceptors for subjective, opinion-based, or comparative placement queries to present neutral, factual answers."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.eligibility_data: List[Dict[str, Any]] = []
        self._load_data()
        logger.info("OpinionGuard initialized")
        
    def _load_data(self) -> None:
        """Load eligibility data for factual backings."""
        eligibility_path = os.path.join(self.data_dir, "eligibility_data.json")
        if os.path.exists(eligibility_path):
            try:
                with open(eligibility_path, 'r') as f:
                    self.eligibility_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading eligibility data in OpinionGuard: {e}")
                
    def execute(self, query: str) -> str:
        """Check the query, compare companies factually if two or more are mentioned, and return a neutral response.
        
        Args:
            query: Career or ranking query
            
        Returns:
            Balanced, neutral explanation
        """
        logger.info(f"OpinionGuard handling query: {query}")
        
        query_lower = query.lower()
        
        # Extract mentioned companies
        mentioned_companies = []
        for company_profile in self.eligibility_data:
            company_name = company_profile["company"].lower()
            if company_name in query_lower:
                mentioned_companies.append(company_profile)
                
        # If two or more companies are mentioned, do a neutral comparison
        if len(mentioned_companies) >= 2:
            return self._generate_neutral_comparison(mentioned_companies)
            
        # If only one company is mentioned
        if len(mentioned_companies) == 1:
            comp = mentioned_companies[0]
            return f"""### Career Advice: Considering **{comp['company']}**

Choosing a company depends heavily on your individual career goals, technical interests, and personal circumstances. Rather than making a subjective ranking, here are the objective facts for **{comp['company']}** from our official records:

* **Package Offered:** {comp.get('package_lpa')} LPA
* **Minimum CGPA Requirement:** {comp.get('min_cgpa')}
* **Maximum Allowable Backlogs:** {comp.get('max_backlogs')}
* **Employment Bond:** {comp.get('bond_years')} Years
* **Technical Focus:** {comp.get('tech_focus')}
* **Key Interview Topics:** {comp.get('key_topics')}

**Recommendation:** If you prioritize {comp.get('tech_focus')} and are comfortable with a {comp.get('bond_years')}-year bond for a package of {comp.get('package_lpa')} LPA, {comp['company']} is a highly structured choice. Compare this with other options to see which aligns best with your goals.
"""

        # General subjective query
        return """### Factual Career Selection Framework

As an AI Assistant, I provide objective data to help you make informed decisions, rather than prescribing subjective career choices or opinions on which company is "better."

When comparing companies in this placement dataset, we recommend evaluating them across these four key vectors:
1. **Financial Package (LPA):** Service-based companies range from 4 to 26 LPA, while product-based firms range from 12 to 43 LPA in this cohort.
2. **Commitment (Bond Years):** A bond-free company (0 years) provides high flexibility, whereas a 2-year bond offers job security but limits short-term transitions.
3. **Cutoffs (CGPA & Backlogs):** Ensure your current academic standing meets the specific cutoff criteria before applying.
4. **Technology Focus:** Align your preparation with the technical topics (e.g. Java, Python, C++, System Design) prioritized by the firm.

*Use the search or database filters to inspect specific parameters.*
"""

    def _generate_neutral_comparison(self, companies: List[Dict[str, Any]]) -> str:
        """Generate side-by-side comparison tables with neutral commentary."""
        names = [c["company"] for c in companies]
        logger.info(f"OpinionGuard comparing factually: {names}")
        
        md = f"### ⚖️ Factual Comparison: {', '.join(names[:-1])} and {names[-1]}\n\n"
        md += "To help you make an objective choice, here is a factual side-by-side comparison from the official placement documents:\n\n"
        
        md += "| Parameter | " + " | ".join([f"**{c['company']}**" for c in companies]) + " |\n"
        md += "| :--- | " + " | ".join([":---:" for _ in companies]) + " |\n"
        
        md += "| **Package (LPA)** | " + " | ".join([f"{c['package_lpa']} LPA" for c in companies]) + " |\n"
        md += "| **Min CGPA Cutoff** | " + " | ".join([str(c['min_cgpa']) for c in companies]) + " |\n"
        md += "| **Max Backlogs** | " + " | ".join([str(c['max_backlogs']) for c in companies]) + " |\n"
        md += "| **Bond Period** | " + " | ".join([f"{c['bond_years']} Years" for c in companies]) + " |\n"
        md += "| **Tech Focus** | " + " | ".join([c['tech_focus'] for c in companies]) + " |\n"
        md += "| **Key Topics** | " + " | ".join([c['key_topics'] for c in companies]) + " |\n"
        
        md += """
### Balanced Insights:
* **Compensation vs Cutoff:** Compare the package-to-CGPA-cutoff ratio. For example, some companies offer high compensation packages with lower CGPA requirements, but may have more rigorous technical rounds.
* **Flexibility vs Security:** A bond-free role offers greater mobility to switch careers, whereas a company with a 1 or 2-year bond offers immediate employment stability.
* **Technical Match:** Choose the firm whose tech focus (e.g., Python vs Java vs System Design) aligns closely with your existing strengths and long-term learning goals.

*Subjective decisions like "which is better" should be based on your personal preference for the parameters listed above.*
"""
        return md
