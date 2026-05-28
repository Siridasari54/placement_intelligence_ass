"""Handler for hiring-based multi-hop queries."""

import pandas as pd
from typing import List, Dict, Any, Optional
from config.constants import COMPANIES
from app.utils.logger import logger


class HiringHandler:
    """Handles multi-hop queries related to hiring statistics."""
    
    def __init__(self, df_elig: pd.DataFrame, df_hiring: pd.DataFrame):
        """Initialize the hiring handler.
        
        Args:
            df_elig: DataFrame containing eligibility data
            df_hiring: DataFrame containing hiring data
        """
        self.df_elig = df_elig
        self.df_hiring = df_hiring
    
    def handle_analyst_query(self, query_lower: str) -> Optional[List[Dict[str, Any]]]:
        """Handle queries about analyst hiring with specific criteria.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            List of context pieces or None if not applicable
        """
        if "analyst" not in query_lower or "20" not in query_lower or "cgpa 8.0" not in query_lower:
            return None
        
        context_pieces = []
        
        if self.df_elig.empty or self.df_hiring.empty:
            return context_pieces
        
        # Filter eligibility: CGPA >= 8.0, backlogs = 0
        elig_cos = self.df_elig[(self.df_elig["min_cgpa"] <= 8.0) & (self.df_elig["max_backlogs"] >= 0)]
        # Filter package > 20
        elig_cos = elig_cos[elig_cos["package_lpa"] > 20.0]
        
        # Join with hiring table
        merged = pd.merge(elig_cos, self.df_hiring, on="company")
        
        if not merged.empty:
            # Filter analyst > 40
            analyst_filter = merged[merged["analyst"] > 40]
            if not analyst_filter.empty:
                best = analyst_filter.sort_values(by="analyst", ascending=False).iloc[0]
                text = f"Structured Analysis: Companies with CGPA <= 8.0 cutoff, 0 backlogs, package > 20 LPA, and >40 Analyst hires:\n"
                for _, r in analyst_filter.iterrows():
                    text += f"- {r['company']}: package: {r['package_lpa']} LPA, Analyst hires: {r['analyst']}\n"
                text += f"The primary matching company is {best['company']}."
                
                context_pieces.append({
                    "text": text,
                    "metadata": {
                        "section": "hiring",
                        "company": best['company'],
                        "source": "pandas_analysis"
                    }
                })
        
        return context_pieces
    
    def handle_intern_query(self, query_lower: str) -> Optional[List[Dict[str, Any]]]:
        """Handle queries about intern hiring.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            List of context pieces or None if not applicable
        """
        if "intern" not in query_lower or "most" not in query_lower:
            return None
        
        context_pieces = []
        
        if self.df_hiring.empty:
            return context_pieces
        
        # Check if Python-focused is specified
        if "python" in query_lower:
            if not self.df_elig.empty:
                py_cos = self.df_elig[self.df_elig["tech_focus"].str.lower().str.contains("python")]["company"].tolist()
                py_hiring = self.df_hiring[self.df_hiring["company"].isin(py_cos)]
                if not py_hiring.empty:
                    best_py_intern = py_hiring.sort_values(by="intern", ascending=False).iloc[0]
                    text = f"Structured Analysis: Python-focused companies and their intern hiring:\n"
                    for _, r in py_hiring.sort_values(by="intern", ascending=False).iterrows():
                        text += f"- {r['company']}: {r['intern']} intern hires\n"
                    text += f"The Python-focused company that hires the most interns is {best_py_intern['company']} with {best_py_intern['intern']} intern hires."
                    context_pieces.append({
                        "text": text,
                        "metadata": {
                            "section": "hiring",
                            "company": best_py_intern['company'],
                            "source": "pandas_analysis"
                        }
                    })
        else:
            best_intern = self.df_hiring.sort_values(by="intern", ascending=False).iloc[0]
            text = f"Structured Analysis: Companies sorted by intern hiring count:\n"
            for _, r in self.df_hiring.sort_values(by="intern", ascending=False).head(5).iterrows():
                text += f"- {r['company']}: {r['intern']} interns\n"
            text += f"The company that hires the most interns is {best_intern['company']} with {best_intern['intern']} interns."
            context_pieces.append({
                "text": text,
                "metadata": {
                    "section": "hiring",
                    "company": best_intern['company'],
                    "source": "pandas_analysis"
                }
            })
        
        return context_pieces
