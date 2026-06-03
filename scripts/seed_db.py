"""Seed SQLite Database with student placement records and company profiles."""

import os
import json
import sqlite3
import random

def seed_database():
    db_path = "data/placement_system.db"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect and create tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create companies table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            company TEXT PRIMARY KEY,
            min_cgpa REAL,
            max_backlogs INTEGER,
            package_lpa REAL,
            bond_years INTEGER,
            key_topics TEXT,
            tech_focus TEXT
        )
    """)
    
    # Create students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            roll_number TEXT PRIMARY KEY,
            name TEXT,
            branch TEXT,
            cgpa REAL,
            backlogs INTEGER,
            placed_company TEXT,
            package_lpa REAL,
            skills TEXT,
            FOREIGN KEY(placed_company) REFERENCES companies(company)
        )
    """)
    
    # Load and seed companies
    eligibility_path = "data/processed/eligibility_data.json"
    companies_data = []
    if os.path.exists(eligibility_path):
        with open(eligibility_path, "r") as f:
            companies_data = json.load(f)
            
        for c in companies_data:
            cursor.execute("""
                INSERT OR REPLACE INTO companies (company, min_cgpa, max_backlogs, package_lpa, bond_years, key_topics, tech_focus)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                c["company"],
                c["min_cgpa"],
                c["max_backlogs"],
                c["package_lpa"],
                c["bond_years"],
                c["key_topics"],
                c["tech_focus"]
            ))
        print(f"Seeded {len(companies_data)} companies in SQL database.")
    else:
        print("Error: eligibility_data.json not found. Companies table empty.")
        return
        
    # Generate 50 realistic student records
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
    
    # Seed generator for reproducibility
    random.seed(42)
    
    students_data = []
    for i in range(1, 51):
        roll_number = f"22CS{i:03d}"
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        branch = random.choice(branches)
        cgpa = round(random.uniform(6.0, 9.8), 2)
        backlogs = random.choice([0, 0, 0, 0, 0, 1, 1, 2, 3]) # Bias towards 0 backlogs
        
        # Pick 3 to 5 random skills
        skills_list = random.sample(all_skills, random.randint(3, 5))
        skills = ", ".join(skills_list)
        
        placed_company = None
        package_lpa = None
        
        # Determine placement status - check eligibility first!
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
            
        students_data.append((
            roll_number,
            name,
            branch,
            cgpa,
            backlogs,
            placed_company,
            package_lpa,
            skills
        ))
        
    cursor.executemany("""
        INSERT OR REPLACE INTO students (roll_number, name, branch, cgpa, backlogs, placed_company, package_lpa, skills)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, students_data)
    
    conn.commit()
    print(f"Seeded {len(students_data)} students in SQL database.")
    
    # Quick verification
    cursor.execute("SELECT COUNT(*) FROM students")
    student_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM companies")
    company_count = cursor.fetchone()[0]
    print(f"Verification: DB has {student_count} students and {company_count} companies.")
    
    conn.close()

if __name__ == "__main__":
    seed_database()
