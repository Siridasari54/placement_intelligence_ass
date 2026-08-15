"""Database service layer offering reusable database utility functions for student and company records."""

import logging
from typing import List, Dict, Any, Optional
from core.database.mysql_manager import MySQLConnectionManager

logger = logging.getLogger(__name__)

class MySQLPlacementService:
    """Service class containing reusable utilities to query, update, and manage the student placement tables in MySQL."""
    
    def __init__(self):
        
        self.manager = MySQLConnectionManager()

    def create_database_and_schema(self) -> bool:
        """Create the database tables and establish primary/foreign key relationships in MySQL."""
        try:
            # Table 1: companies
            create_companies_query = """
            CREATE TABLE IF NOT EXISTS companies (
                company VARCHAR(100) PRIMARY KEY,
                min_cgpa DOUBLE,
                max_backlogs INT,
                package_lpa DOUBLE,
                bond_years INT,
                key_topics TEXT,
                tech_focus TEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            # Table 2: students
            create_students_query = """
            CREATE TABLE IF NOT EXISTS students (
                roll_number VARCHAR(50) PRIMARY KEY,
                name VARCHAR(150),
                branch VARCHAR(50),
                cgpa DOUBLE,
                backlogs INT,
                placed_company VARCHAR(100),
                package_lpa DOUBLE,
                skills TEXT,
                FOREIGN KEY (placed_company) REFERENCES companies(company) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            logger.info("Verifying/creating MySQL tables...")
            self.manager.execute_query(create_companies_query, is_select=False)
            self.manager.execute_query(create_students_query, is_select=False)
            logger.info("MySQL tables ('companies', 'students') verified/created successfully.")
            return True
        except Exception as e:
            logger.error(f"Error creating MySQL database schema: {e}", exc_info=True)
            return False

    def insert_company(self, company_data: Dict[str, Any]) -> bool:
        """Insert or replace a company record using parameterized queries."""
        query = """
        INSERT INTO companies (company, min_cgpa, max_backlogs, package_lpa, bond_years, key_topics, tech_focus)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            min_cgpa = VALUES(min_cgpa),
            max_backlogs = VALUES(max_backlogs),
            package_lpa = VALUES(package_lpa),
            bond_years = VALUES(bond_years),
            key_topics = VALUES(key_topics),
            tech_focus = VALUES(tech_focus)
        """
        params = (
            company_data["company"],
            company_data["min_cgpa"],
            company_data["max_backlogs"],
            company_data["package_lpa"],
            company_data["bond_years"],
            company_data["key_topics"],
            company_data["tech_focus"]
        )
        try:
            self.manager.execute_query(query, params, is_select=False)
            return True
        except Exception as e:
            logger.error(f"Failed to upsert company '{company_data.get('company')}': {e}")
            return False

    def insert_student(self, student_data: Dict[str, Any]) -> bool:
        """Insert or replace a student record using parameterized queries."""
        query = """
        INSERT INTO students (roll_number, name, branch, cgpa, backlogs, placed_company, package_lpa, skills)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            name = VALUES(name),
            branch = VALUES(branch),
            cgpa = VALUES(cgpa),
            backlogs = VALUES(backlogs),
            placed_company = VALUES(placed_company),
            package_lpa = VALUES(package_lpa),
            skills = VALUES(skills)
        """
        params = (
            student_data["roll_number"],
            student_data["name"],
            student_data["branch"],
            student_data["cgpa"],
            student_data["backlogs"],
            student_data["placed_company"],
            student_data["package_lpa"],
            student_data["skills"]
        )
        try:
            self.manager.execute_query(query, params, is_select=False)
            return True
        except Exception as e:
            logger.error(f"Failed to upsert student '{student_data.get('roll_number')}': {e}")
            return False

    def get_student_by_roll(self, roll_number: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific student's placement record by roll number."""
        query = "SELECT * FROM students WHERE roll_number = %s"
        try:
            res = self.manager.execute_query(query, (roll_number,), is_select=True)
            return res[0] if res else None
        except Exception as e:
            logger.error(f"Error querying student with roll number '{roll_number}': {e}")
            return None

    def get_placed_students_by_company(self, company_name: str) -> List[Dict[str, Any]]:
        """Fetch all students placed at a specific company."""
        query = "SELECT * FROM students WHERE LOWER(placed_company) = LOWER(%s) ORDER BY cgpa DESC"
        try:
            return self.manager.execute_query(query, (company_name,), is_select=True)
        except Exception as e:
            logger.error(f"Error fetching placed students for company '{company_name}': {e}")
            return []

    def get_companies_matching_cutoff(self, cgpa: float, backlogs: int) -> List[Dict[str, Any]]:
        """Get all companies where a student with the given CGPA and backlog count is eligible."""
        query = "SELECT * FROM companies WHERE min_cgpa <= %s AND max_backlogs >= %s ORDER BY package_lpa DESC"
        try:
            return self.manager.execute_query(query, (cgpa, backlogs), is_select=True)
        except Exception as e:
            logger.error(f"Error querying eligible companies: {e}")
            return []

    def run_custom_parameterized_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Run any custom SELECT query with parameters securely."""
        try:
            return self.manager.execute_query(query, params, is_select=True)
        except Exception as e:
            logger.error(f"Error executing custom query: {e}")
            raise
