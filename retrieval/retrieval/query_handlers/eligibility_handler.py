"""Handler for eligibility-based multi-hop queries."""

import re
import pandas as pd
from typing import List, Dict, Any, Optional
from config.constants import IT_SERVICE_FIRMS
from app.utils.logger import logger


class EligibilityHandler:
    """Handles multi-hop queries related to eligibility criteria."""
    
    def __init__(self, df_elig: pd.DataFrame):
        """Initialize the eligibility handler.
        
        Args:
            df_elig: DataFrame containing eligibility data
        """
        self.df_elig = df_elig
    
    def handle_cgpa_backlog_query(self, query_lower: str) -> Optional[List[Dict[str, Any]]]:
        """Handle queries with CGPA and backlog criteria.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            List of context pieces or None if not applicable
        """
        if "backlog" not in query_lower or ("cgpa" not in query_lower and "gpa" not in query_lower):
            return None
        
        context_pieces = []
        cgpa_val = 7.6
        backlog_val = 1
        bond_free_only = False
        
        # Extract numbers from query
        cgpa_matches = re.findall(r"cgpa\s+(?:of\s+)?(\d+\.\d+|\d+)", query_lower)
        backlog_matches = re.findall(r"(\d+)\s+backlog", query_lower)
        
        if cgpa_matches:
            cgpa_val = float(cgpa_matches[0])
        if backlog_matches:
            backlog_val = int(backlog_matches[0])
        
        if "no bond" in query_lower or "bond-free" in query_lower or "bond free" in query_lower or "without bond" in query_lower:
            bond_free_only = True
        
        if self.df_elig.empty:
            return context_pieces
        
        # Filter companies
        filtered = self.df_elig[
            (self.df_elig["min_cgpa"] <= cgpa_val) & 
            (self.df_elig["max_backlogs"] >= backlog_val)
        ]
        
        if bond_free_only:
            filtered = filtered[filtered["bond_years"] == 0]
        
        if not filtered.empty:
            sorted_df = filtered.sort_values(by="package_lpa", ascending=False)
            best_match = sorted_df.iloc[0]
            
            text = (
                f"Structured Analysis Result: Students with CGPA {cgpa_val} and {backlog_val} backlog(s) "
                f"{'and no bond' if bond_free_only else ''} qualify for the following companies (sorted by package):\n"
            )
            for _, r in sorted_df.iterrows():
                text += f"- {r['company']}: Package: {r['package_lpa']} LPA, CGPA Cutoff: {r['min_cgpa']}, Max Backlogs: {r['max_backlogs']}, Bond: {r['bond_years']} Yrs.\n"
            
            text += f"The highest-paying company they qualify for is {best_match['company']} at {best_match['package_lpa']} LPA."
            
            context_pieces.append({
                "text": text,
                "metadata": {
                    "section": "eligibility",
                    "company": best_match['company'],
                    "source": "pandas_analysis"
                }
            })
        
        return context_pieces
    
    def handle_python_query(self, query_lower: str) -> Optional[List[Dict[str, Any]]]:
        """Handle queries about Python-focused companies.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            List of context pieces or None if not applicable
        """
        if "python" not in query_lower or "highest package" not in query_lower:
            return None
        
        context_pieces = []
        
        if self.df_elig.empty:
            return context_pieces
        
        py_cos = self.df_elig[
            (self.df_elig["tech_focus"].str.lower().str.contains("python")) |
            (self.df_elig["key_topics"].str.lower().str.contains("python"))
        ]
        
        if not py_cos.empty:
            sorted_py = py_cos.sort_values(by="package_lpa", ascending=False)
            best_py = sorted_py.iloc[0]
            text = "Structured Analysis: Python-focused companies and packages:\n"
            for _, r in sorted_py.iterrows():
                text += f"- {r['company']}: {r['package_lpa']} LPA\n"
            text += f"The Python-focused company offering the highest package is {best_py['company']} at {best_py['package_lpa']} LPA."
            
            context_pieces.append({
                "text": text,
                "metadata": {
                    "section": "eligibility",
                    "company": best_py['company'],
                    "source": "pandas_analysis"
                }
            })
        
        return context_pieces
    
    def handle_zero_bond_query(self, query_lower: str) -> Optional[List[Dict[str, Any]]]:
        """Handle queries about zero-bond companies.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            List of context pieces or None if not applicable
        """
        if "zero-bond" not in query_lower and "zero bond" not in query_lower and "bond-free" not in query_lower and "no bond" not in query_lower:
            return None
        
        context_pieces = []
        
        if "40" not in query_lower and "highest package" not in query_lower:
            return context_pieces
        
        if self.df_elig.empty:
            return context_pieces
        
        bond_free_40 = self.df_elig[(self.df_elig["bond_years"] == 0) & (self.df_elig["package_lpa"] > 40.0)]
        
        if not bond_free_40.empty:
            text = "Structured Analysis: Zero-bond companies offering more than 40 LPA:\n"
            for _, r in bond_free_40.iterrows():
                text += f"- {r['company']}: {r['package_lpa']} LPA\n"
            context_pieces.append({
                "text": text,
                "metadata": {
                    "section": "eligibility",
                    "source": "pandas_analysis"
                }
            })
        
        return context_pieces
    
    def handle_service_firm_query(self, query_lower: str) -> Optional[List[Dict[str, Any]]]:
        """Handle queries about IT service firms.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            List of context pieces or None if not applicable
        """
        if "highest package" not in query_lower or ("service firm" not in query_lower and "service companies" not in query_lower):
            return None
        
        context_pieces = []
        
        if self.df_elig.empty:
            return context_pieces
        
        service_df = self.df_elig[self.df_elig["company"].isin(IT_SERVICE_FIRMS)]
        
        if not service_df.empty:
            best_service = service_df.sort_values(by="package_lpa", ascending=False).iloc[0]
            text = "Structured Analysis: IT Service firms sorted by package:\n"
            for _, r in service_df.sort_values(by="package_lpa", ascending=False).iterrows():
                text += f"- {r['company']}: {r['package_lpa']} LPA\n"
            text += f"The IT service firm with the highest package is {best_service['company']} at {best_service['package_lpa']} LPA."
            
            context_pieces.append({
                "text": text,
                "metadata": {
                    "section": "eligibility",
                    "company": best_service['company'],
                    "source": "pandas_analysis"
                }
            })
        
        return context_pieces
    
    def handle_package_ratio_query(self, query_lower: str) -> Optional[List[Dict[str, Any]]]:
        """Handle queries about package-to-CGPA ratio.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            List of context pieces or None if not applicable
        """
        if "package-to-cgpa ratio" not in query_lower and "package to cgpa" not in query_lower:
            return None
        
        context_pieces = []
        
        if self.df_elig.empty:
            return context_pieces
        
        df_ratio = self.df_elig.copy()
        df_ratio["ratio"] = df_ratio["package_lpa"] / df_ratio["min_cgpa"]
        sorted_ratio = df_ratio.sort_values(by="ratio", ascending=False)
        best_ratio = sorted_ratio.iloc[0]
        
        text = "Structured Analysis: Package-to-CGPA cut-off ratios (LPA per CGPA point):\n"
        for _, r in sorted_ratio.head(5).iterrows():
            text += f"- {r['company']}: Package: {r['package_lpa']} LPA, CGPA: {r['min_cgpa']}, Ratio: {r['ratio']:.2f}\n"
        text += f"The company offering the best package-to-CGPA ratio is {best_ratio['company']} with a ratio of {best_ratio['ratio']:.2f}."
        
        context_pieces.append({
            "text": text,
            "metadata": {
                "section": "eligibility",
                "company": best_ratio['company'],
                "source": "pandas_analysis"
            }
        })
        
        return context_pieces
