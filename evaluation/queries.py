"""Evaluation queries for automated testing of the RAG system."""

# 30 official placement queries covering different aspects
OFFICIAL_QUERIES = [
    # Eligibility queries
    "What is the minimum CGPA required for Google?",
    "What is the maximum backlogs allowed for Amazon?",
    "What is the package offered by Microsoft?",
    "What is the service bond period for Adobe?",
    "What is the tech focus for Uber?",
    
    # Interview experience queries
    "What are the interview rounds for Meta?",
    "What are the preparation tips for Apple?",
    "What is the technical focus for Netflix?",
    "How many rounds does Tesla have?",
    "What are the tips for NVIDIA interviews?",
    
    # Hiring distribution queries
    "How many SDE roles does Google offer?",
    "How many analyst roles does Amazon offer?",
    "How many intern roles does Microsoft offer?",
    "What is the total hiring for Adobe?",
    "How many officer roles does Uber offer?",
    
    # Temporal trend queries
    "What was the package offered by Google in 2021?",
    "What was the package offered by Amazon in 2022?",
    "What is the 3-year trend for Microsoft?",
    "What was the package in 2023 for Adobe?",
    "What is the trend direction for Uber?",
    
    # Statistics queries
    "What is the average package for Google?",
    "What is the maximum offers for Amazon?",
    "What is the minimum offers for Microsoft?",
    "What is the average CGPA cutoff for Adobe?",
    "Is Adobe bond-free?",
    
    # General queries
    "Which companies have the highest package?",
    "Which companies allow the most backlogs?",
    "What is the most common tech focus?",
    "Which companies have the longest bond?",
    "Which companies have the most hiring?"
]

# 4 multi-hop queries requiring reasoning
MULTIHOP_QUERIES = [
    "Which company offers the best package for students with 8.5 CGPA and no backlogs?",
    "Compare the interview processes of Google and Amazon",
    "Which companies have improved their packages from 2021 to 2024?",
    "What are the eligibility requirements for top 3 companies by package?"
]

# Combined query set
EVALUATION_QUERIES = OFFICIAL_QUERIES + MULTIHOP_QUERIES
