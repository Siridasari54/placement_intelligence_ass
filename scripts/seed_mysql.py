"""Seed XAMPP phpMyAdmin MySQL Database with student placement records and company profiles."""

import os
import sys
import json
import random

# Add root directory to path to enable clean imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database.mysql_service import MySQLPlacementService
from core.database.mysql_manager import MySQLConnectionManager

def seed_mysql_database():
    print("Initializing MySQL Placement Service...")
    service = MySQLPlacementService()
    
    # Step 1: Create Database & Schema
    print("Creating database and schema if not exists...")
    schema_ok = service.create_database_and_schema()
    if not schema_ok:
        print("Error: Could not bootstrap MySQL database schema. Please check if XAMPP MySQL is running.")
        return False
        
    # Step 2: Load Company Data from eligibility_data.json
    eligibility_path = "data/processed/eligibility_data.json"
    companies_data = []
    
    if os.path.exists(eligibility_path):
        with open(eligibility_path, "r") as f:
            companies_data = json.load(f)
            
        print(f"Loaded {len(companies_data)} companies from eligibility_data.json.")
        
        # Insert companies into MySQL
        seeded_companies_count = 0
        for c in companies_data:
            success = service.insert_company({
                "company": c["company"],
                "min_cgpa": c["min_cgpa"],
                "max_backlogs": c["max_backlogs"],
                "package_lpa": c["package_lpa"],
                "bond_years": c["bond_years"],
                "key_topics": c["key_topics"],
                "tech_focus": c["tech_focus"]
            })
            if success:
                seeded_companies_count += 1
        print(f"Seeded {seeded_companies_count} companies in MySQL database.")
    else:
        print("Error: eligibility_data.json not found. Cannot seed companies. Exiting.")
        return False
        
    # Step 3: Generate 50 realistic student records (exact parity with SQLite seeder)
    first_names = [
        "Aarav", "Vihaan", "Aditya", "Sai", "Arjun", "Krishna", "Ishaan", "Shaurya", "Pranav", "Aryan",
        "Ananya", "Diya", "Saanvi", "Aadhya", "Prisha", "Riya", "Ira", "Avani", "Kavya", "Myra",
        "Rahul", "Rohan", "Siddharth", "Varun", "Karan", "Sneha", "Neha", "Priya", "Anjali", "Pooja"
    ]
    last_names = [
        "Sharma", "Verma", "Gupta", "Kumar", "Singh", "Reddy", "Patel", "Mehra", "Joshi", "Rao",
        "Iyer", "Nair", "Das", "Choudhury", "Banerjee", "Chatterjee", "Mishra", "Pandey", "Saxena", "Deshmukh"
    ]
    branches = ["CSE", "ECE", "EEE", "MECH", "CIVIL", "IT"]
    all_skills = ["Python", "Java", "C++", "SQL", "HTML/CSS", "JavaScript", "Machine Learning", "System Design", "Cloud Computing", "OOPs", "DBMS"]
    
    # Seed generator for complete reproducibility
    random.seed(42)
    
    students_data = []
    seeded_students_count = 0
    
    for i in range(1, 51):
        roll_number = f"22CS{i:03d}"
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        branch = random.choice(branches)
        cgpa = round(random.uniform(6.0, 9.8), 2)
        backlogs = random.choice([0, 0, 0, 0, 0, 1, 1, 2, 3])  # Bias towards 0 backlogs
        
        # Pick 3 to 5 random skills
        skills_list = random.sample(all_skills, random.randint(3, 5))
        skills = ", ".join(skills_list)
        
        placed_company = None
        package_lpa = None
        
        # Filter companies where student meets eligibility
        eligible_companies = [
            c for c in companies_data 
            if cgpa >= c["min_cgpa"] and backlogs <= c["max_backlogs"]
        ]
        
        # 60% chance of being placed if there are eligible companies
        if eligible_companies and random.random() < 0.6:
            chosen_comp = random.choice(eligible_companies)
            placed_company = chosen_comp["company"]
            package_lpa = chosen_comp["package_lpa"]
            
        student_record = {
            "roll_number": roll_number,
            "name": name,
            "branch": branch,
            "cgpa": cgpa,
            "backlogs": backlogs,
            "placed_company": placed_company,
            "package_lpa": package_lpa,
            "skills": skills
        }
        
        success = service.insert_student(student_record)
        if success:
            seeded_students_count += 1
            
    print(f"Seeded {seeded_students_count} students in MySQL database.")
    
    # Step 4: Verification
    try:
        student_check = service.run_custom_parameterized_query("SELECT COUNT(*) as count FROM students")
        company_check = service.run_custom_parameterized_query("SELECT COUNT(*) as count FROM companies")
        
        student_count = student_check[0]["count"] if student_check else 0
        company_count = company_check[0]["count"] if company_check else 0
        
        print("\n=============================================")
        print("MYSQL DATABASE SEEDING COMPLETED SUCCESSFULLY!")
        print(f"Verification: DB has {student_count} students and {company_count} companies.")
        print("=============================================")
        return True
    except Exception as verify_err:
        print(f"Verification query failed: {verify_err}")
        return False

if __name__ == "__main__":
    seed_mysql_database()
