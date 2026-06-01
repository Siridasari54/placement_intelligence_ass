"""Database tool for structured lookup and filtering using SQLite and LLM Text-to-SQL translation."""

import os
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq
from config.settings import settings

logger = logging.getLogger(__name__)

class DatabaseTool:
    """Uses LLM (Groq) to translate natural language questions into SQL queries, executes them against SQLite database, and returns results."""
    
    def __init__(self, data_dir: str = "data"):
        self.db_path = os.path.join(data_dir, "placement_system.db")
        self.api_key = settings.groq_api_key
        self.model = settings.generation.model
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Error initializing Groq client in DatabaseTool: {e}")
        logger.info("DatabaseTool (Text-to-SQL) initialized")
        
    def execute(self, query: str) -> str:
        """Parse natural language query to SQL, run it, and return a formatted markdown table.
        
        Args:
            query: User's natural language question
            
        Returns:
            Markdown table or error message
        """
        logger.info(f"DatabaseTool processing query: {query}")
        
        if not os.path.exists(self.db_path):
            return "No placement database exists. Please run the database seeder first."
            
        if not self.client:
            # Fallback if Groq client is not available (e.g. key missing)
            return "SQL database is available, but the AI routing key is missing. Cannot translate natural language to SQL."
            
        # 1. Generate SQL query using LLM
        sql_query, explanation = self._generate_sql(query)
        if not sql_query:
            return "I could not formulate a database query for your question."
            
        logger.info(f"Generated SQL: {sql_query}")
        
        # 2. Execute SQL query
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            # 3. Format results as Markdown
            return self._format_results(sql_query, columns, rows, explanation)
            
        except Exception as e:
            logger.error(f"SQL Execution error on '{sql_query}': {e}")
            return f"### 📊 SQL Database Query\nI generated the following SQL query but it failed to run:\n```sql\n{sql_query}\n```\n*Error: {str(e)}*"

    def _generate_sql(self, question: str) -> Tuple[Optional[str], str]:
        """Generate SQLite SQL query from natural language question."""
        schema_info = """
Table 1: companies
- company TEXT (Primary Key, e.g. 'TCS', 'Google', 'Amazon')
- min_cgpa REAL (Minimum CGPA cutoff)
- max_backlogs INTEGER (Maximum backlogs allowed)
- package_lpa REAL (Package offered in LPA)
- bond_years INTEGER (Bond period in years)
- key_topics TEXT (Key interview topics)
- tech_focus TEXT (Core tech focus, e.g. 'Java', 'Python', 'System Design')

Table 2: students
- roll_number TEXT (Primary Key, e.g. '22CS001')
- name TEXT (Student name)
- branch TEXT (Branch, e.g. 'CSE', 'ECE', 'MECH', 'IT')
- cgpa REAL (Student CGPA)
- backlogs INTEGER (Current number of backlogs)
- placed_company TEXT (Foreign Key to companies.company, NULL if unplaced)
- package_lpa REAL (Package package in LPA, NULL if unplaced)
- skills TEXT (Comma-separated list of skills, e.g. 'Python, SQL, OOPs')
"""
        
        prompt = f"""You are a precise SQLite SQL query generator.
Given the following database schema, write a valid SQLite SELECT query to answer the user's question.

Schema:
{schema_info}

Instructions:
- Respond in a JSON format containing two keys:
  - "sql": The exact SQLite query string. It must contain ONLY the SQL statement. No code fences, no extra text, no newlines inside string.
  - "explanation": A brief, 1-sentence description of what the query is looking up.
- Never write destructive queries (INSERT, UPDATE, DELETE, DROP). Only SELECT is allowed.
- Use case-insensitive matching where applicable (e.g. LOWER(placed_company) = 'tcs' or placed_company LIKE '%tcs%').
- Do not use markdown syntax or markdown blocks inside the JSON output.
- Limit query results to 15 records unless specifically asked for more.

Question: {question}

JSON response:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a database tool that outputs JSON containing 'sql' and 'explanation'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            import json
            data = json.loads(content)
            
            sql = data.get("sql", "").strip()
            explanation = data.get("explanation", "")
            
            # Clean up SQL if LLM added markdown formatting
            if sql.startswith("```"):
                sql = sql.split("```")[1]
                if sql.startswith("sql"):
                    sql = sql[3:]
            sql = sql.strip(";").strip()
            
            return sql, explanation
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return None, ""

    def _format_results(self, sql: str, columns: List[str], rows: List[Tuple], explanation: str) -> str:
        """Format SQL result list into a Markdown table."""
        if not rows:
            return f"### 📊 Database Lookup\n*Query Intent: {explanation}*\n\nNo student or company records matched your criteria in the database."
            
        md = f"### 📊 Database Lookup\n*Query Intent: {explanation}*\n\n"
        
        # Headers
        md += "| " + " | ".join(columns) + " |\n"
        md += "| " + " | ".join([":---:" if isinstance(r, (int, float)) else ":---" for r in rows[0]]) + " |\n"
        
        # Rows
        for row in rows:
            formatted_cells = []
            for cell in row:
                if cell is None:
                    formatted_cells.append("*None*")
                elif isinstance(cell, float):
                    formatted_cells.append(f"{cell:.2f}")
                else:
                    formatted_cells.append(str(cell))
            md += "| " + " | ".join(formatted_cells) + " |\n"
            
        md += f"\n\n<details><summary>🔍 View Generated SQL</summary>\n\n```sql\n{sql}\n```\n</details>"
        return md
