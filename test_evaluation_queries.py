"""Test script for evaluation queries from the RAG-ATHON 24 dataset."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import RAGPipeline
from core.di.factories import register_services
from core.di.container import ServiceContainer

# Easy queries (E1-E10)
EASY_QUERIES = [
    "What is the CGPA requirement for TCS?",
    "How many backlogs does Deloitte allow?",
    "What is the bond period for Amazon?",
    "Which technology does Flipkart focus on in interviews?",
    "What is the package offered by Google?",
    "Does Microsoft allow backlogs?",
    "What rounds does TCS conduct?",
    "Which programming language is tested at Amazon?"
]

# Medium queries (M1-M10)
MEDIUM_QUERIES = [
    "List all companies that allow at least 2 backlogs.",
    "Which companies require a CGPA above 8.0?",
    "Which company has the highest package among IT service firms?",
    "Which companies are bond-free?",
    "Compare TCS and Infosys on all eligibility criteria.",
    "How many SDE roles does Amazon hire versus Google?",
    "Which company hires the most Interns?",
    "What topics should I prepare for a Microsoft interview?",
    "Which company's package grew the most from 2021 to 2024?",
    "Which companies use Python as the technical focus?"
]

# Hard queries (H1-H7)
HARD_QUERIES = [
    "A student with CGPA 7.0, 1 backlog wants maximum pay with no bond.",
    "Which Python-focused company hires the most Interns?",
    "For CGPA 8.0+, zero backlog students, rank companies by package.",
    "Which company had conflicting CGPA data across sources?",
    "Is the Amazon CGPA cutoff 6.4 or 7.0? Explain.",
    "Which company offers the best package-to-CGPA ratio?",
    "Compare Google and Amazon on all dimensions: eligibility, package, hiring, trend."
]

# Expert queries (X1-X5)
EXPERT_QUERIES = [
    "What is TCS's campus visit date at SVECW?",
    "Should I join Google or Microsoft? Which is better for my career?",
    "I have CGPA 5.0. Where can I apply?",
    "What is Infosys's current stock price?",
    "Which company in this dataset pays the highest in the world?"
]


def test_queries(queries, category):
    """Test a set of queries and print results."""
    print(f"\n{'='*80}")
    print(f"Testing {category} Queries")
    print(f"{'='*80}")
    
    # Initialize pipeline
    container = ServiceContainer()
    register_services(container)
    pipeline = container.get_service(RAGPipeline)
    
    if not pipeline:
        print("ERROR: Pipeline not initialized")
        return
    
    results = []
    for idx, query in enumerate(queries, 1):
        print(f"\n[{idx}] Query: {query}")
        try:
            result = pipeline.query(query)
            answer = result.get("answer", "No answer")
            confidence = result.get("confidence", 0.0)
            sources = result.get("sources", [])
            
            print(f"Answer: {answer[:200]}...")
            print(f"Confidence: {confidence:.2f}")
            print(f"Sources: {len(sources)}")
            
            results.append({
                "query": query,
                "answer": answer,
                "confidence": confidence,
                "sources_count": len(sources)
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "query": query,
                "answer": f"ERROR: {str(e)}",
                "confidence": 0.0,
                "sources_count": 0
            })
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test evaluation queries")
    parser.add_argument("--category", choices=["easy", "medium", "hard", "expert", "all"], 
                        default="easy", help="Category of queries to test")
    args = parser.parse_args()
    
    if args.category == "easy" or args.category == "all":
        test_queries(EASY_QUERIES, "Easy")
    
    if args.category == "medium" or args.category == "all":
        test_queries(MEDIUM_QUERIES, "Medium")
    
    if args.category == "hard" or args.category == "all":
        test_queries(HARD_QUERIES, "Hard")
    
    if args.category == "expert" or args.category == "all":
        test_queries(EXPERT_QUERIES, "Expert")


if __name__ == "__main__":
    main()
