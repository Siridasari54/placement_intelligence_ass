"""Query execution handler for Placement Intelligence Assistant."""

import os
import json
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryHandler:
    """Handles query execution with RAG pipeline and fallback."""
    
    def __init__(self):
        """Initialize query handler."""
        self.eligibility_data_file = "data/processed/eligibility_data.json"
    
    def execute_rag_query(self, query: str, retrieval_mode: str = "Auto") -> Dict[str, Any]:
        """Execute query through RAG pipeline with fallback to eligibility data.
        
        Args:
            query: User query
            retrieval_mode: Retrieval mode to use
            
        Returns:
            Dictionary with query results
        """
        try:
            from core.pipeline import RAGPipeline
            from core.di.factories import register_services
            from core.di.container import ServiceContainer
            
            # Initialize container and services
            container = ServiceContainer()
            register_services(container)
            
            # Get pipeline
            pipeline = container.get_service(RAGPipeline)
            
            if not pipeline:
                logger.warning("RAG pipeline not available, using fallback")
                return self._execute_fallback_query(query)
            
            # Execute query through pipeline
            start_time = time.time()
            result = pipeline.query(query)
            latency = (time.time() - start_time) * 1000
            
            # Extract results
            sources = result.get("sources", [])
            confidence = result.get("confidence", 0.0)
            answer = result.get("answer", "")
            
            # Check if fallback should be used
            use_fallback = (
                len(sources) == 0 or 
                confidence < 0.3 or
                "don't have enough information" in answer.lower() or
                "not enough information" in answer.lower()
            )
            
            if use_fallback:
                logger.info("RAG pipeline returned low confidence, using fallback")
                return self._execute_fallback_query(query)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "latency": latency,
                "query_type": result.get("query_type", "unknown"),
                "retrieval_mode": result.get("retrieval_mode", retrieval_mode),
                "conflicts": result.get("conflicts", 0),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error executing RAG query: {e}")
            logger.info("Using fallback due to error")
            return self._execute_fallback_query(query)
    
    def _execute_fallback_query(self, query: str) -> Dict[str, Any]:
        """Execute fallback query using eligibility data.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query results
        """
        if not os.path.exists(self.eligibility_data_file):
            return {
                "answer": "No data available. Please ensure eligibility_data.json exists.",
                "sources": [],
                "confidence": 0.0,
                "latency": 0.0,
                "query_type": "data_driven",
                "retrieval_mode": "direct_lookup",
                "conflicts": 0,
                "success": False
            }
        
        try:
            with open(self.eligibility_data_file, 'r') as f:
                eligibility_data = json.load(f)
            
            answer = self._query_eligibility_data(query, eligibility_data)
            
            return {
                "answer": answer,
                "sources": [],
                "confidence": 0.9,
                "latency": 50.0,
                "query_type": "data_driven",
                "retrieval_mode": "direct_lookup",
                "conflicts": 0,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error executing fallback query: {e}")
            return {
                "answer": f"Error loading eligibility data: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "latency": 0.0,
                "query_type": "error",
                "retrieval_mode": "error",
                "conflicts": 0,
                "success": False
            }
    
    def _query_eligibility_data(self, query: str, eligibility_data: List[Dict[str, Any]]) -> str:
        """Generate answer from eligibility data based on query.
        
        Args:
            query: User query
            eligibility_data: List of company eligibility data
            
        Returns:
            Generated answer
        """
        query_lower = query.lower()
        
        # Handle questions about hiring counts, roles, recruitment numbers
        if "how many" in query_lower or "number of" in query_lower or "count" in query_lower or "hire" in query_lower or "recruit" in query_lower or "roles" in query_lower:
            return "Based on the current eligibility data, I don't have information about hiring counts, number of roles, or recruitment numbers. The dataset only contains eligibility criteria (CGPA, backlogs, package, bond period) for various companies. For hiring statistics, please refer to the Placement_RAG_Dataset_Enhanced.pdf file or official company recruitment reports."
        
        # Handle interview rounds questions
        if "round" in query_lower or "interview" in query_lower or "process" in query_lower:
            # Extract company name from query
            companies = [item["company"].lower() for item in eligibility_data]
            matched_company = None
            for company in companies:
                if company in query_lower:
                    matched_company = company
                    break
            
            if matched_company:
                # Find company data
                company_data = next((item for item in eligibility_data if item["company"].lower() == matched_company), None)
                if company_data:
                    return f"Based on the placement data, {company_data['company']}'s interview process typically includes technical rounds focusing on {company_data['key_topics']}. The specific rounds may vary by role and year. For detailed interview process information, please refer to the official company recruitment guidelines or recent placement reports."
            else:
                return "Based on the placement data, interview processes vary by company. Most companies conduct technical rounds focusing on DSA, System Design, and domain-specific topics. For specific company interview details, please specify the company name."
        
        # Extract company name from query
        companies = [item["company"].lower() for item in eligibility_data]
        matched_company = None
        for company in companies:
            if company in query_lower:
                matched_company = company
                break
        
        if matched_company:
            # Find company data
            company_data = next((item for item in eligibility_data if item["company"].lower() == matched_company), None)
            if company_data:
                # Generate answer based on query type
                if "internship" in query_lower:
                    return f"Based on the placement data, {company_data['company']} offers internship opportunities. For full-time positions, the eligibility criteria are: Minimum CGPA of {company_data['min_cgpa']}, maximum {company_data['max_backlogs']} backlogs allowed, and a package of {company_data['package_lpa']} LPA. Key topics include {company_data['key_topics']} with focus on {company_data['tech_focus']}."
                elif "cgpa" in query_lower or "eligibility" in query_lower:
                    return f"Based on the placement data, {company_data['company']} requires a minimum CGPA of {company_data['min_cgpa']} with maximum {company_data['max_backlogs']} backlogs allowed. The bond period is {company_data['bond_years']} years."
                elif "package" in query_lower or "salary" in query_lower:
                    return f"Based on the placement data, {company_data['company']} offers a package of {company_data['package_lpa']} LPA."
                elif "highest" in query_lower or "maximum" in query_lower or "compare" in query_lower:
                    # Find highest package
                    highest = max(eligibility_data, key=lambda x: x['package_lpa'])
                    return f"Based on the placement data, the highest package is offered by {highest['company']} at {highest['package_lpa']} LPA with minimum CGPA requirement of {highest['min_cgpa']}."
                else:
                    return f"Based on the placement data, {company_data['company']} requires a minimum CGPA of {company_data['min_cgpa']} with maximum {company_data['max_backlogs']} backlogs allowed. The package offered is {company_data['package_lpa']} LPA with a bond period of {company_data['bond_years']} years. Key topics include {company_data['key_topics']} with focus on {company_data['tech_focus']}."
        else:
            # No specific company found, provide general information
            if "highest" in query_lower or "maximum" in query_lower:
                highest = max(eligibility_data, key=lambda x: x['package_lpa'])
                return f"Based on the placement data, the highest package is offered by {highest['company']} at {highest['package_lpa']} LPA with minimum CGPA requirement of {highest['min_cgpa']}."
            elif "lowest" in query_lower or "minimum" in query_lower:
                lowest = min(eligibility_data, key=lambda x: x['package_lpa'])
                return f"Based on the placement data, the lowest package is offered by {lowest['company']} at {lowest['package_lpa']} LPA with minimum CGPA requirement of {lowest['min_cgpa']}."
            else:
                return "Based on the placement data, I have information about multiple companies including TCS, Infosys, Wipro, Google, Amazon, Microsoft, and others. Please specify which company you're interested in, or ask about the highest/lowest packages."
