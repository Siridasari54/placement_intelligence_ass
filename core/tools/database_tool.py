"""Database tool for structured lookup and filtering using MySQL and LLM Text-to-SQL translation with SQLite fallback."""

import os
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq
from config.settings import settings
from core.database.mysql_manager import MySQLConnectionManager

logger = logging.getLogger(__name__)

class DatabaseTool:
    """Uses LLM (Groq) to translate natural language questions into SQL queries, executes them against MySQL, and returns results.
    
    If the MySQL database is offline (e.g. XAMPP not running), it gracefully falls back to SQLite.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.sqlite_db_path = os.path.join(data_dir, "placement_system.db")
        self.api_key = settings.groq_api_key
        self.model = settings.generation.model
        self.client = None
        self.mysql_manager = None
        
        # Initialize Groq client
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Error initializing Groq client in DatabaseTool: {e}")
                
        # Initialize MySQL Connection Manager
        try:
            self.mysql_manager = MySQLConnectionManager()
            logger.info("DatabaseTool MySQL connection manager initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize MySQL Connection Manager in DatabaseTool: {e}")
            
        logger.info("DatabaseTool (Dual Engine: MySQL + SQLite fallback) initialized")
        
    def execute(self, query: str) -> str:
        """Parse natural language query to SQL, run it on MySQL (or SQLite fallback), and return a formatted markdown table.
        
        Args:
            query: User's natural language question
            
        Returns:
            Markdown table or error message
        """
        logger.info(f"DatabaseTool processing query: {query}")
        
        if not self.client:
            return "SQL database is available, but the AI routing key is missing. Cannot translate natural language to SQL."
            
        # 1. Generate SQL query using LLM
        sql_query, explanation = self._generate_sql(query)
        if not sql_query:
            return "I could not formulate a database query for your question."
            
        logger.info(f"Generated SQL: {sql_query}")
        
        # 2. Execute SQL query - Attempt MySQL first
        mysql_failed = False
        mysql_error = ""
        columns = []
        rows = []
        
        if self.mysql_manager:
            try:
                logger.info("Attempting to execute query on MySQL...")
                # Run query via connection pool
                raw_rows = self.mysql_manager.execute_query(sql_query, is_select=True)
                
                if raw_rows:
                    columns = list(raw_rows[0].keys())
                    rows = [tuple(row.values()) for row in raw_rows]
                else:
                    # For empty results, try fetching column names from description
                    conn = self.mysql_manager.get_connection()
                    cursor = conn.cursor(dictionary=True)
                    cursor.execute(sql_query)
                    columns = [desc[0] for desc in cursor.description]
                    cursor.close()
                    conn.close()
                    rows = []
                
                logger.info(f"Successfully executed query on MySQL. Retrieved {len(rows)} records.")
                # Format results as Markdown
                return self._format_results(sql_query, columns, rows, explanation, database_engine="MySQL")
                
            except Exception as mysql_err:
                mysql_failed = True
                mysql_error = str(mysql_err)
                logger.warning(f"MySQL execution failed: {mysql_err}. Falling back to SQLite.")
        else:
            mysql_failed = True
            mysql_error = "MySQL connection manager is offline."
            
        # 3. SQLite Fallback Execution
        if mysql_failed:
            if not os.path.exists(self.sqlite_db_path):
                return (f"### 📊 SQL Database Query\nMySQL execution failed (*{mysql_error}*).\n"
                        "No SQLite fallback database exists. Please seed the database first.")
                        
            try:
                logger.info("Executing query on SQLite fallback database...")
                conn = sqlite3.connect(self.sqlite_db_path)
                cursor = conn.cursor()
                
                # SQLite sometimes syntax check fails for MySQL dialect specific constructs,
                # but our prompt generates standard ANSI queries.
                cursor.execute(sql_query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                conn.close()
                
                logger.info(f"Successfully executed query on SQLite fallback. Retrieved {len(rows)} records.")
                return self._format_results(sql_query, columns, rows, explanation, database_engine="SQLite (Fallback)")
            except Exception as sqlite_err:
                logger.error(f"SQLite fallback query failed: {sqlite_err}")
                return (f"### 📊 SQL Database Query\nI generated the following SQL query but it failed to run on both MySQL and SQLite:\n"
                        f"```sql\n{sql_query}\n```\n*MySQL Error: {mysql_error}*\n*SQLite Error: {str(sqlite_err)}*")

    def _generate_sql(self, question: str) -> Tuple[Optional[str], str]:
        """Generate MySQL SQL query from natural language question."""
        schema_info = """
Table 1: companies
- company VARCHAR(100) (Primary Key, e.g. 'TCS', 'Google', 'Amazon')
- min_cgpa DOUBLE (Minimum CGPA cutoff)
- max_backlogs INT (Maximum backlogs allowed)
- package_lpa DOUBLE (Package offered in LPA)
- bond_years INT (Bond period in years)
- key_topics TEXT (Key interview topics)
- tech_focus TEXT (Core tech focus, e.g. 'Java', 'Python', 'System Design')

Table 2: students
- roll_number VARCHAR(50) (Primary Key, e.g. '22CS001')
- name VARCHAR(150) (Student name)
- branch VARCHAR(50) (Branch, e.g. 'CSE', 'ECE', 'MECH', 'IT')
- cgpa DOUBLE (Student CGPA)
- backlogs INT (Current number of backlogs)
- placed_company VARCHAR(100) (Foreign Key to companies.company, NULL if unplaced)
- package_lpa DOUBLE (Package package in LPA, NULL if unplaced)
- skills TEXT (Comma-separated list of skills, e.g. 'Python, SQL, OOPs')
"""
        
        prompt = f"""You are a precise MySQL SQL query generator.
Given the following database schema, write a valid ANSI-compatible MySQL SELECT query to answer the user's question.

Schema:
{schema_info}

Instructions:
- Respond in a JSON format containing two keys:
  - "sql": The exact MySQL query string. It must contain ONLY the SQL statement. No code fences, no extra text, no newlines inside string.
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

    def _format_results(self, sql: str, columns: List[str], rows: List[Tuple], explanation: str, database_engine: str) -> str:
        """Format SQL result list into a Markdown table."""
        if not rows:
            return (f"### 📊 Database Lookup ({database_engine})\n"
                    f"*Query Intent: {explanation}*\n\n"
                    f"No student or company records matched your criteria in the database.")
            
        md = f"### 📊 Database Lookup ({database_engine})\n*Query Intent: {explanation}*\n\n"
        
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
