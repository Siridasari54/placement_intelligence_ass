import re

class QueryRewriter:
    def __init__(self):
        # Maps user terms to standard corpus terms
        self.synonyms = {
            r"\bpay\b": "package",
            r"\bsalary\b": "package",
            r"\bcompensation\b": "package",
            r"\bcutoff\b": "CGPA requirement",
            r"\bcut-off\b": "CGPA requirement",
            r"\bbonds\b": "bond years",
            r"\bexperience\b": "interview experience",
            r"\bno bond\b": "bond free 0 years",
            r"\bzero bond\b": "bond free 0 years",
            r"\bzero backlog\b": "0 backlogs",
            r"\bno backlogs\b": "0 backlogs",
        }

    def rewrite(self, query: str) -> str:
        """Standardizes query terminology."""
        rewritten = query.lower()
        for pattern, replacement in self.synonyms.items():
            rewritten = re.sub(pattern, replacement, rewritten)
            
        # Clean double spacing
        rewritten = " ".join(rewritten.split())
        return rewritten
