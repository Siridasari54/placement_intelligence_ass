"""Calculator tool for safe arithmetic, CGPA, and package calculations."""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CalculatorTool:
    """Performs precise arithmetic calculations, CGPA-to-percentage conversions, and statistical computations."""
    
    def __init__(self):
        logger.info("CalculatorTool initialized")
        
    def execute(self, query: str) -> str:
        """Route the query to the appropriate calculation method and execute.
        
        Args:
            query: Math or conversion query string
            
        Returns:
            Calculated result as a string
        """
        logger.info(f"CalculatorTool executing query: {query}")
        
        query_lower = query.lower()
        
        # 1. CGPA to Percentage Conversion
        if "cgpa to percentage" in query_lower or "convert" in query_lower and "cgpa" in query_lower:
            return self._convert_cgpa_to_percentage(query)
            
        # 2. Percentage to CGPA Conversion
        if "percentage to cgpa" in query_lower:
            return self._convert_percentage_to_cgpa(query)
            
        # 3. Average Package / Statistics Calculation
        if "average" in query_lower or "mean" in query_lower:
            return self._calculate_average(query)
            
        # 4. Standard Arithmetic
        return self._evaluate_arithmetic(query)

    def _convert_cgpa_to_percentage(self, query: str) -> str:
        """Convert CGPA to percentage using standard Indian university conversion formula (CGPA * 9.5)."""
        numbers = re.findall(r'\b\d+\.?\d*\b', query)
        if not numbers:
            return "No CGPA value detected in query. Please specify a value, e.g., 'Convert 8.5 CGPA to percentage'."
            
        cgpa = float(numbers[0])
        if cgpa < 0.0 or cgpa > 10.0:
            return f"Invalid CGPA value: {cgpa}. CGPA must be between 0.0 and 10.0."
            
        percentage = cgpa * 9.5
        return f"According to standard college placement guidelines, a CGPA of {cgpa} converts to **{percentage:.2f}%** (Formula: CGPA * 9.5)."

    def _convert_percentage_to_cgpa(self, query: str) -> str:
        """Convert percentage to CGPA (Percentage / 9.5)."""
        numbers = re.findall(r'\b\d+\.?\d*\b', query)
        if not numbers:
            return "No percentage value detected in query. Please specify a value, e.g., 'Convert 85% to CGPA'."
            
        percentage = float(numbers[0])
        if percentage < 0.0 or percentage > 100.0:
            return f"Invalid percentage value: {percentage}%. Percentage must be between 0% and 100%."
            
        cgpa = percentage / 9.5
        return f"A percentage of {percentage}% converts to a CGPA of **{cgpa:.2f}** out of 10 (Formula: Percentage / 9.5)."

    def _calculate_average(self, query: str) -> str:
        """Calculate average of a sequence of numbers (e.g. packages)."""
        numbers = [float(x) for x in re.findall(r'\b\d+\.?\d*\b', query)]
        if not numbers:
            return "No numbers found to calculate average."
            
        avg = sum(numbers) / len(numbers)
        return f"The calculated average (mean) is **{avg:.2f}** (calculated from {len(numbers)} values: {', '.join(map(str, numbers))})."

    def _evaluate_arithmetic(self, query: str) -> str:
        """Evaluate standard arithmetic expressions safely."""
        # Strip query down to mathematical characters
        expression = re.sub(r'[^0-9+\-*/().\s]', '', query).strip()
        if not expression:
            return "Could not extract a valid mathematical expression from query."
            
        try:
            # Safe evaluation using restricted builtins
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                raise ValueError("Unsafe characters detected in expression")
                
            # Perform calculation
            result = eval(expression, {"__builtins__": None}, {})
            return f"Mathematical Calculation: `{expression}` = **{result}**"
        except Exception as e:
            logger.error(f"Arithmetic evaluation failed for '{expression}': {e}")
            return f"I apologize, but I could not compute that mathematical expression: {str(e)}"
