# Constants for Placement Intelligence Assistant

# Companies list
COMPANIES = [
    "TCS", "Infosys", "Deloitte", "Accenture", "Amazon", "Flipkart", 
    "Google", "Microsoft", "Wipro", "Cognizant", "Capgemini", "IBM", 
    "Adobe", "Oracle", "SAP", "HCL", "Tech Mahindra", "Qualcomm", 
    "Intel", "Samsung R&D"
]

# Classification for queries comparing categories (e.g. M3: Highest package among IT service firms)
IT_SERVICE_FIRMS = [
    "TCS", "Infosys", "Deloitte", "Accenture", "Wipro", 
    "Cognizant", "Capgemini", "HCL", "Tech Mahindra"
]

PRODUCT_COMPANIES = [
    "Amazon", "Flipkart", "Google", "Microsoft", "IBM", 
    "Adobe", "Oracle", "SAP", "Qualcomm", "Intel", "Samsung R&D"
]

# Companies with known conflicting data
CONFLICT_COMPANIES = ["TCS", "Amazon", "Google", "Infosys", "Microsoft"]

# Global Configurations (Replacing YAML configuration files)
MODEL_CONFIG = {
    "llm": {
        "provider": "groq",
        "default_model": "llama3-70b-8192",
        "alternative_model": "mixtral-8x7b-32768",
        "temperature": 0.1,
        "max_tokens": 1024,
        "timeout": 30.0
    },
    "embeddings": {
        "provider": "huggingface",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu"
    }
}

RETRIEVAL_CONFIG = {
    "hybrid": {
        "weight_vector": 0.6,
        "weight_keyword": 0.4,
        "top_k": 10
    },
    "bm25": {
        "k1": 1.5,
        "b": 0.75
    },
    "reranker": {
        "enabled": True,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_n": 5
    },
    "filtering": {
        "allow_metadata_filtering": True,
        "default_min_relevance_score": 0.65
    }
}

CHUNKING_CONFIG = {
    "general": {
        "chunk_size": 500,
        "chunk_overlap": 50
    },
    "strategies": {
        "company_profiles": {
            "mode": "row-per-company",
            "chunk_size": 1
        },
        "interview_experiences": {
            "mode": "semantic",
            "chunk_size": 300,
            "chunk_overlap": 30
        },
        "statistics": {
            "mode": "full-table"
        },
        "temporal_trends": {
            "mode": "row-per-company-per-year"
        },
        "conflicting_records": {
            "mode": "keep-both-flag-conflict"
        }
    }
}

CACHE_CONFIG = {
    "cache": {
        "enabled": True,
        "type": "semantic",
        "db_path": "data/cache/semantic_cache.db",
        "similarity_threshold": 0.88
    }
}
