import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from app.utils.logger import logger
from config.constants import COMPANIES


class TemporalRetriever:
    """Handles time-indexed trend queries and performs mathematical calculations."""
    
    def __init__(self, processed_dir: str = "data/processed"):
        """Initialize the temporal retriever.
        
        Args:
            processed_dir: Directory containing processed data files
        """
        self.processed_dir = processed_dir

    def _load_trend_data(self) -> Optional[pd.DataFrame]:
        """Load trend data from JSON file.
        
        Returns:
            DataFrame with trend data if successful, None otherwise
        """
        path = os.path.join(self.processed_dir, "trend_data.json")
        if not os.path.exists(path):
            logger.warning(f"Trend data file not found: {path}")
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            
            if df.empty:
                logger.warning("Trend data is empty")
                return None
            
            return df
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in trend_data.json: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading trend data: {e}")
            return None

    def _detect_growth_query(self, query_lower: str) -> bool:
        """Check if query is about growth/maximum increase.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            True if query is about growth, False otherwise
        """
        growth_keywords = [
            "grew the most",
            "largest absolute",
            "largest increase",
            "maximum growth",
            "highest growth"
        ]
        return any(keyword in query_lower for keyword in growth_keywords)

    def _process_growth_query(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process queries about package growth.
        
        Args:
            df: DataFrame with trend data
            
        Returns:
            List of context pieces with growth analysis
        """
        context_pieces = []
        
        try:
            df["increase"] = df["package_2024"] - df["package_2021"]
            sorted_df = df.sort_values(by="increase", ascending=False)
            best = sorted_df.iloc[0]
            
            text = "Structured Temporal Analysis: Absolute Package Growth from 2021 to 2024:\n"
            for _, r in sorted_df.iterrows():
                text += f"- {r['company']}: grew by {r['increase']:.1f} LPA (from {r['package_2021']} to {r['package_2024']} LPA)\n"
            text += f"\nThe company that showed the largest absolute package increase from 2021 to 2024 is {best['company']} which grew by {best['increase']:.1f} LPA (from {best['package_2021']} to {best['package_2024']} LPA)."
            
            context_pieces.append({
                "text": text,
                "metadata": {
                    "section": "trend",
                    "company": best["company"],
                    "source": "pandas_temporal"
                }
            })
        except Exception as e:
            logger.error(f"Error processing growth query: {e}")
        
        return context_pieces

    def _process_company_trend_query(self, df: pd.DataFrame, query_lower: str) -> List[Dict[str, Any]]:
        """Process queries about specific company trends.
        
        Args:
            df: DataFrame with trend data
            query_lower: Lowercase query string
            
        Returns:
            List of context pieces with trend analysis
        """
        context_pieces = []
        
        try:
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
                        "metadata": {
                            "section": "trend",
                            "source": "pandas_temporal"
                        }
                    })
        except Exception as e:
            logger.error(f"Error processing company trend query: {e}")
        
        return context_pieces

    def retrieve_temporal_trend(self, query: str) -> List[Dict[str, Any]]:
        """Handles time-indexed trend queries and performs mathematical calculations.
        
        Args:
            query: User query string
            
        Returns:
            List of context pieces with temporal analysis
        """
        query_lower = query.lower()
        context_pieces = []
        
        df = self._load_trend_data()
        if df is None:
            return context_pieces
        
        # Process growth queries
        if self._detect_growth_query(query_lower):
            context_pieces.extend(self._process_growth_query(df))
        else:
            # Process general trend queries
            context_pieces.extend(self._process_company_trend_query(df, query_lower))
        
        return context_pieces
