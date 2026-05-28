"""Handler for company comparison queries."""

import pandas as pd
from typing import List, Dict, Any, Optional
from config.constants import COMPANIES


class ComparisonHandler:
    """Handles multi-hop queries for comparing companies."""
    
    def __init__(self, df_elig: pd.DataFrame, df_hiring: pd.DataFrame, df_trend: pd.DataFrame):
        """Initialize the comparison handler.
        
        Args:
            df_elig: DataFrame containing eligibility data
            df_hiring: DataFrame containing hiring data
            df_trend: DataFrame containing trend data
        """
        self.df_elig = df_elig
        self.df_hiring = df_hiring
        self.df_trend = df_trend
    
    def handle_comparison_query(self, query_lower: str) -> Optional[List[Dict[str, Any]]]:
        """Handle queries comparing specific companies.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            List of context pieces or None if not applicable
        """
        # Extract companies mentioned in query
        compare_companies = []
        for c in COMPANIES:
            if c.lower() in query_lower:
                compare_companies.append(c)
        
        if len(compare_companies) < 2:
            return None
        
        context_pieces = []
        comparison_text = f"Structured Analysis comparison of {', '.join(compare_companies)}:\n"
        
        # Eligibility
        if not self.df_elig.empty:
            elig_sub = self.df_elig[self.df_elig["company"].isin(compare_companies)]
            comparison_text += "\nEligibility & Cutoff Criteria:\n"
            comparison_text += elig_sub.to_string(index=False) + "\n"
        
        # Hiring
        if not self.df_hiring.empty:
            hiring_sub = self.df_hiring[self.df_hiring["company"].isin(compare_companies)]
            comparison_text += "\nHiring Role Distribution:\n"
            comparison_text += hiring_sub.to_string(index=False) + "\n"
        
        # Trend
        if not self.df_trend.empty:
            trend_sub = self.df_trend[self.df_trend["company"].isin(compare_companies)]
            comparison_text += "\nYearly Packages Trend:\n"
            comparison_text += trend_sub.to_string(index=False) + "\n"
        
        context_pieces.append({
            "text": comparison_text,
            "metadata": {
                "section": "comparison",
                "source": "pandas_analysis"
            }
        })
        
        return context_pieces
