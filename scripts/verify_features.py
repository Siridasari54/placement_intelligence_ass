"""Programmatic verification of the newly added agentic and UI-supporting features."""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tools.web_search import WebSearchTool
from core.tools.database_tool import DatabaseTool
from core.tools.resume_analyzer import ResumeAnalyzer
from core.pipeline import ToolRouter

def test_web_search():
    print("\n--- Testing Web Search Tool ---")
    tool = WebSearchTool()
    # Live fetch Google's current CEO
    res = tool.execute("Google CEO")
    print(f"Web Search Length: {len(res)} characters")
    assert "Google" in res, "Web search result should contain company context"
    print("Web Search Tool: SUCCESS")

def test_database_tool():
    print("\n--- Testing Text-to-SQL Database Tool ---")
    tool = DatabaseTool()
    # Lookup top 3 student profiles
    res = tool.execute("Find the top 3 students by CGPA")
    print(f"Database Result Length: {len(res)} characters")
    assert "Database" in res or "sql" in res.lower(), "Should produce a valid database report"
    print("Database Tool (SQL): SUCCESS")

def test_resume_analyzer():
    print("\n--- Testing Resume Gap Analyzer ---")
    analyzer = ResumeAnalyzer()
    
    mock_resume = """
    John Doe
    Computer Science Student
    Skills: Python, SQL, HTML, CSS, JavaScript, Git.
    Projects: E-commerce Website using Django and React.
    Looking for SDE positions.
    CGPA: 8.5, Backlogs: 0.
    """
    
    res = analyzer.analyze_resume(mock_resume)
    print(f"Resume Report Length: {len(res)} characters")
    assert len(res) > 100, "Should generate a substantial analysis report"
    print("Resume Match Analyzer: SUCCESS")

def test_tool_router():
    print("\n--- Testing LLM-based Tool Router ---")
    router = ToolRouter()
    
    # Register tools
    from core.tools.calculator import CalculatorTool
    from core.tools.opinion_guard import OpinionGuard
    
    router.register_tool("calculator", CalculatorTool())
    router.register_tool("database", DatabaseTool())
    router.register_tool("opinion_guard", OpinionGuard())
    router.register_tool("web_search", WebSearchTool())
    
    # 1. Test database query classification
    db_res = router.classify_and_dispatch("Show me the student with the highest GPA")
    print(f"Database dispatch length: {len(db_res) if db_res else 0}")
    assert db_res is not None, "Should dispatch highest GPA query to database tool"
    
    # 2. Test calculator query classification
    calc_res = router.classify_and_dispatch("What is 8.5 CGPA in percentage?")
    print(f"Calculator dispatch result: {calc_res}")
    assert calc_res is not None, "Should dispatch CGPA-to-percentage query to calculator tool"
    
    # 3. Test opinion guard query classification
    op_res = router.classify_and_dispatch("Should I join Google or Amazon?")
    print(f"Opinion Guard dispatch length: {len(op_res) if op_res else 0}")
    assert op_res is not None, "Should dispatch subjective choice to opinion guard"
    
    # 4. Test RAG query classification
    rag_res = router.classify_and_dispatch("Explain SVECW campus interview rounds")
    print(f"RAG query classification: {rag_res}")
    assert rag_res is None, "RAG-focused questions should not route to any specific tool (should return None for RAG pipeline)"
    
    print("LLM Tool Router: SUCCESS")

if __name__ == "__main__":
    print("Starting Placement Assistant Verification...")
    try:
        test_web_search()
        test_database_tool()
        test_resume_analyzer()
        test_tool_router()
        print("\n===============================")
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("===============================")
    except Exception as e:
        print(f"\nVerification FAILED: {str(e)}")
        sys.exit(1)
