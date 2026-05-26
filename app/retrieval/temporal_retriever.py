import os
import json
import pandas as pd
from typing import List, Dict, Any
from app.utils.logger import logger

class TemporalRetriever:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir

    def retrieve_temporal_trend(self, query: str) -> List[Dict[str, Any]]:
        """Handles time-indexed trend queries and performs mathematical calculations."""
        query_lower = query.lower()
        context_pieces = []
        
        path = os.path.join(self.processed_dir, "trend_data.json")
        if not os.path.exists(path):
            return []
            
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        if df.empty:
            return []
            
        # Case 1: "package grew the most" or "largest absolute package increase"
        if "grew the most" in query_lower or "largest absolute" in query_lower or "largest increase" in query_lower or "maximum growth" in query_lower:
            df["increase"] = df["package_2024"] - df["package_2021"]
            sorted_df = df.sort_values(by="increase", ascending=False)
            best = sorted_df.iloc[0]
            
            text = "Structured Temporal Analysis: Absolute Package Growth from 2021 to 2024:\n"
            for _, r in sorted_df.iterrows():
                text += f"- {r['company']}: grew by {r['increase']:.1f} LPA (from {r['package_2021']} to {r['package_2024']} LPA)\n"
            text += f"\nThe company that showed the largest absolute package increase from 2021 to 2024 is {best['company']} which grew by {best['increase']:.1f} LPA (from {best['package_2021']} to {best['package_2024']} LPA)."
            
            context_pieces.append({
                "text": text,
                "metadata": {"section": "trend", "company": best["company"], "source": "pandas_temporal"}
            })
            
        # Case 2: General trend fetching
        # e.g., "compare google and infosys package trends"
        else:
            # Check if specific companies are mentioned
            from app.utils.constants import COMPANIES
            matched_companies = [c for c in COMPANIES if c.lower() in query_lower]
            
            if matched_companies:
                filtered = df[df["company"].isin(matched_companies)]
                if not filtered.empty:
                    text = "Structured Temporal Analysis: Package Trends (2021-2024):\n"
                    for _, r in filtered.iterrows():
                        text += (
                            f"- {r['company']}: 2021: {r['package_2021']} LPA, 2022: {r['package_2022']} LPA, "
                            f"2023: {r['package_2023']} LPA, 2024: {r['package_2024']} LPA. "
                            f"Trend: {r['trend_direction']}\n"
                        )
                    context_pieces.append({
                        "text": text,
                        "metadata": {"section": "trend", "source": "pandas_temporal"}
                    })
                    
        return context_pieces
