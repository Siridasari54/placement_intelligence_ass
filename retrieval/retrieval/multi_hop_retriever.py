import os
import json
import pandas as pd
from typing import List, Dict, Any
from app.utils.logger import logger
from app.retrieval.query_handlers import EligibilityHandler, HiringHandler, ComparisonHandler


class MultiHopRetriever:
    """Retriever for multi-hop queries using specialized handlers."""
    
    def __init__(self, processed_dir: str = "data/processed"):
        """Initialize the multi-hop retriever.
        
        Args:
            processed_dir: Directory containing processed data files
        """
        self.processed_dir = processed_dir

    def load_table(self, name: str) -> pd.DataFrame:
        """Loads a processed JSON file as a Pandas DataFrame.
        
        Args:
            name: Name of the data file (without _data.json suffix)
            
        Returns:
            DataFrame with loaded data or empty DataFrame if file not found
        """
        path = os.path.join(self.processed_dir, f"{name}_data.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            except Exception as e:
                logger.error(f"Error loading {name}_data.json: {e}")
                return pd.DataFrame()
        logger.warning(f"Processed file not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()

    def retrieve_multi_hop(self, query: str) -> List[Dict[str, Any]]:
        """Analyzes multi-hop queries and performs structural joins on JSON data.
        
        Args:
            query: User query string
            
        Returns:
            List of context pieces with analysis results
        """
        query_lower = query.lower()
        context_pieces = []
        
        # Load tables
        df_elig = self.load_table("eligibility")
        df_hiring = self.load_table("hiring")
        df_trend = self.load_table("trend")
        
        # Initialize handlers
        eligibility_handler = EligibilityHandler(df_elig)
        hiring_handler = HiringHandler(df_elig, df_hiring)
        comparison_handler = ComparisonHandler(df_elig, df_hiring, df_trend)
        
        # Try each handler
        results = eligibility_handler.handle_cgpa_backlog_query(query_lower)
        if results:
            context_pieces.extend(results)
        
        results = eligibility_handler.handle_python_query(query_lower)
        if results:
            context_pieces.extend(results)
        
        results = eligibility_handler.handle_zero_bond_query(query_lower)
        if results:
            context_pieces.extend(results)
        
        results = eligibility_handler.handle_service_firm_query(query_lower)
        if results:
            context_pieces.extend(results)
        
        results = eligibility_handler.handle_package_ratio_query(query_lower)
        if results:
            context_pieces.extend(results)
        
        results = hiring_handler.handle_analyst_query(query_lower)
        if results:
            context_pieces.extend(results)
        
        results = hiring_handler.handle_intern_query(query_lower)
        if results:
            context_pieces.extend(results)
        
        results = comparison_handler.handle_comparison_query(query_lower)
        if results:
            context_pieces.extend(results)
        
        return context_pieces
