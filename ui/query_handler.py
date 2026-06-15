"""Query execution handler for Placement Intelligence Assistant."""

import os
import json
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryHandler:
    """Handles query execution with intelligent routing to appropriate tools."""
    
    def __init__(self):
        """Initialize query handler with tool router and tools."""
        self.eligibility_data_file = "data/processed/eligibility_data.json"
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize all tools for routing."""
        # Initialize all tool attributes to None first
        self.tool_router = None
        self.web_search_tool = None
        self.database_tool = None
        self.calculator_tool = None
        self.opinion_guard = None
        self.resume_analyzer = None
        
        try:
            from core.pipeline import ToolRouter
            from core.tools.web_search import WebSearchTool
            from core.tools.database_tool import DatabaseTool
            from core.tools.calculator import CalculatorTool
            from core.tools.opinion_guard import OpinionGuard
            from core.tools.resume_analyzer import ResumeAnalyzer
            
            # Initialize tool router
            self.tool_router = ToolRouter()
            
            # Initialize and register tools individually to handle failures
            try:
                self.web_search_tool = WebSearchTool()
                self.tool_router.register_tool("web_search", self.web_search_tool)
                logger.info("WebSearchTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize WebSearchTool: {e}")
            
            try:
                self.database_tool = DatabaseTool()
                self.tool_router.register_tool("database", self.database_tool)
                logger.info("DatabaseTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize DatabaseTool: {e}")
            
            try:
                self.calculator_tool = CalculatorTool()
                self.tool_router.register_tool("calculator", self.calculator_tool)
                logger.info("CalculatorTool initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize CalculatorTool: {e}")
            
            try:
                self.opinion_guard = OpinionGuard()
                self.tool_router.register_tool("opinion_guard", self.opinion_guard)
                logger.info("OpinionGuard initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpinionGuard: {e}")
            
            try:
                self.resume_analyzer = ResumeAnalyzer()
                self.tool_router.register_tool("resume_analyzer", self.resume_analyzer)
                logger.info("ResumeAnalyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ResumeAnalyzer: {e}")
            
            logger.info("QueryHandler initialized with available tools")
        except Exception as e:
            logger.error(f"Error initializing tools in QueryHandler: {e}")
            # Keep all tools as None, will be handled in routing logic
    
    def execute_rag_query(self, query: str, retrieval_mode: str = "Auto") -> Dict[str, Any]:
        """Execute query with intelligent routing to appropriate tool.
        
        Args:
            query: User query
            retrieval_mode: Retrieval mode to use (for RAG only)
            
        Returns:
            Dictionary with query results
        """
        # Step 1: Classify the query to determine the appropriate route
        route = self._classify_query(query)
        logger.info(f"Query classified as: {route}")
        
        # Step 2: Route to appropriate tool based on classification
        if route == "date":
            return self._handle_date_time_query(query)
        elif route == "web_search":
            return self._execute_web_search(query)
        elif route == "database":
            return self._execute_database_query(query)
        elif route == "calculator":
            return self._execute_calculator(query)
        elif route == "opinion":
            return self._execute_opinion_guard(query)
        elif route == "resume":
            return self._execute_resume_analyzer(query)
        elif route == "rag":
            return self._execute_rag_pipeline(query, retrieval_mode)
        else:
            # Default to RAG if classification fails
            logger.warning(f"Unknown route '{route}', defaulting to RAG")
            return self._execute_rag_pipeline(query, retrieval_mode)
    
    def _classify_query(self, query: str) -> str:
        """Classify query to determine appropriate route.
        
        Args:
            query: User query
            
        Returns:
            Route string: "rag", "web_search", "database", "calculator", "opinion", "resume", "date"
        """
        query_lower = query.lower()
        
        # Date/time queries - highest priority
        date_time_indicators = [
            "today", "date", "time", "what day", "what month", "what year",
            "current date", "current time", "now", "what's the date", "what's the time"
        ]
        if any(indicator in query_lower for indicator in date_time_indicators):
            return "date"
        
        # Resume-related queries
        resume_indicators = [
            "resume", "cv", "curriculum vitae", "analyze my resume", "review my resume",
            "upload resume", "check resume"
        ]
        if any(indicator in query_lower for indicator in resume_indicators):
            return "resume"
        
        # Calculator queries
        calculator_indicators = [
            "calculate", "average", "mean", "sum", "convert", "percentage",
            "cgpa to percentage", "percentage to cgpa", "multiply", "divide", "add", "subtract"
        ]
        if any(indicator in query_lower for indicator in calculator_indicators) and any(c.isdigit() for c in query):
            return "calculator"
        
        # Database queries
        database_indicators = [
            "show students", "list students", "top students", "highest cgpa", "lowest cgpa",
            "how many students", "student records", "roll number", "placed students",
            "list companies", "companies with", "package above", "package below"
        ]
        if any(indicator in query_lower for indicator in database_indicators):
            return "database"
        
        # Opinion/subjective queries
        opinion_indicators = [
            "should i join", "which is better", "compare", "versus", "vs",
            "recommend", "advice", "suggestion", "which company", "choose between"
        ]
        if any(indicator in query_lower for indicator in opinion_indicators):
            return "opinion"
        
        # Web search queries (general knowledge, current affairs)
        web_search_indicators = [
            "population", "capital", "who is president", "who is prime minister",
            "latest news", "current affairs", "ipl", "stock market", "weather",
            "temperature", "ceo", "founder", "what is the population", "who won"
        ]
        if any(indicator in query_lower for indicator in web_search_indicators):
            return "web_search"
        
        # Default to RAG for placement-related queries
        return "rag"
    
    def _handle_date_time_query(self, query: str) -> Dict[str, Any]:
        """Handle date/time queries with system information.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with date/time answer
        """
        from datetime import datetime
        
        query_lower = query.lower()
        
        if "date" in query_lower:
            answer = f"Today's date is {datetime.now().strftime('%B %d, %Y')}"
        elif "time" in query_lower:
            answer = f"Current time is {datetime.now().strftime('%I:%M %p')}"
        elif "day" in query_lower:
            answer = f"Today is {datetime.now().strftime('%A')}"
        else:
            answer = f"Today's date is {datetime.now().strftime('%B %d, %Y')} and the time is {datetime.now().strftime('%I:%M %p')}"
        
        return {
            "answer": answer,
            "sources": [],
            "confidence": 1.0,
            "latency": 0.0,
            "query_type": "date_time",
            "retrieval_mode": "system",
            "conflicts": 0,
            "success": True
        }
    
    def _execute_web_search(self, query: str) -> Dict[str, Any]:
        """Execute query through web search tool.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with web search results
        """
        if self.web_search_tool is None:
            logger.warning("WebSearchTool not available, falling back to RAG")
            return self._execute_rag_pipeline(query)
        
        try:
            start_time = time.time()
            answer = self.web_search_tool.execute(query)
            latency = (time.time() - start_time) * 1000
            
            return {
                "answer": answer,
                "sources": [],
                "confidence": 0.8,
                "latency": latency,
                "query_type": "web_search",
                "retrieval_mode": "web_search",
                "conflicts": 0,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error executing web search: {e}")
            return {
                "answer": f"Error performing web search: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "latency": 0.0,
                "query_type": "error",
                "retrieval_mode": "error",
                "conflicts": 0,
                "success": False
            }
    
    def _execute_database_query(self, query: str) -> Dict[str, Any]:
        """Execute query through database tool.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with database results
        """
        if self.database_tool is None:
            logger.warning("DatabaseTool not available, falling back to RAG")
            return self._execute_rag_pipeline(query)
        
        try:
            start_time = time.time()
            answer = self.database_tool.execute(query)
            latency = (time.time() - start_time) * 1000
            
            return {
                "answer": answer,
                "sources": [],
                "confidence": 0.9,
                "latency": latency,
                "query_type": "database",
                "retrieval_mode": "database",
                "conflicts": 0,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error executing database query: {e}")
            return {
                "answer": f"Error querying database: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "latency": 0.0,
                "query_type": "error",
                "retrieval_mode": "error",
                "conflicts": 0,
                "success": False
            }
    
    def _execute_calculator(self, query: str) -> Dict[str, Any]:
        """Execute query through calculator tool.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with calculation results
        """
        if self.calculator_tool is None:
            logger.warning("CalculatorTool not available, falling back to RAG")
            return self._execute_rag_pipeline(query)
        
        try:
            start_time = time.time()
            answer = self.calculator_tool.execute(query)
            latency = (time.time() - start_time) * 1000
            
            return {
                "answer": answer,
                "sources": [],
                "confidence": 1.0,
                "latency": latency,
                "query_type": "calculator",
                "retrieval_mode": "calculator",
                "conflicts": 0,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error executing calculator: {e}")
            return {
                "answer": f"Error performing calculation: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "latency": 0.0,
                "query_type": "error",
                "retrieval_mode": "error",
                "conflicts": 0,
                "success": False
            }
    
    def _execute_opinion_guard(self, query: str) -> Dict[str, Any]:
        """Execute query through opinion guard.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with opinion results
        """
        if self.opinion_guard is None:
            logger.warning("OpinionGuard not available, falling back to RAG")
            return self._execute_rag_pipeline(query)
        
        try:
            start_time = time.time()
            answer = self.opinion_guard.execute(query)
            latency = (time.time() - start_time) * 1000
            
            return {
                "answer": answer,
                "sources": [],
                "confidence": 0.7,
                "latency": latency,
                "query_type": "opinion",
                "retrieval_mode": "opinion",
                "conflicts": 0,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error executing opinion guard: {e}")
            return {
                "answer": f"Error providing opinion: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "latency": 0.0,
                "query_type": "error",
                "retrieval_mode": "error",
                "conflicts": 0,
                "success": False
            }
    
    def _execute_resume_analyzer(self, query: str) -> Dict[str, Any]:
        """Execute query through resume analyzer.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with resume analysis results
        """
        if self.resume_analyzer is None:
            logger.warning("ResumeAnalyzer not available, falling back to RAG")
            return self._execute_rag_pipeline(query)
        
        try:
            start_time = time.time()
            answer = self.resume_analyzer.execute(query)
            latency = (time.time() - start_time) * 1000
            
            return {
                "answer": answer,
                "sources": [],
                "confidence": 0.8,
                "latency": latency,
                "query_type": "resume",
                "retrieval_mode": "resume",
                "conflicts": 0,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error executing resume analyzer: {e}")
            return {
                "answer": f"Error analyzing resume: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "latency": 0.0,
                "query_type": "error",
                "retrieval_mode": "error",
                "conflicts": 0,
                "success": False
            }
    
    def _execute_rag_pipeline(self, query: str, retrieval_mode: str = "Auto") -> Dict[str, Any]:
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
            
            # Check if fallback should be used (do not fallback for tools or cache)
            is_tool_or_cache = result.get("query_type") in ["tool_query", "cached"]
            use_fallback = not is_tool_or_cache and (
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
