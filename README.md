# Placement Intelligence Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Production-Grade Hybrid Multimodal RAG System for Placement Data Analysis**

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This is a production-grade Retrieval-Augmented Generation (RAG) system designed for placement intelligence. It implements a modular 6-stage architecture with hybrid retrieval, multimodal support, safety layers, and continuous feedback loops. The system follows SOLID principles and clean architecture patterns for maintainability and scalability.

### What This Project Does

The Placement Intelligence Assistant helps students and placement coordinators quickly access and analyze placement data through a conversational AI interface. Users can ask natural language questions about:
- Company eligibility criteria (CGPA, backlogs, packages, bonds)
- Interview experiences and preparation tips
- Hiring statistics and trends
- Temporal placement data from 2021-2024

The system provides accurate, context-aware answers with citations from the source documents.

---

## Motivation

### Why I Built This Project

I built this project to solve a common problem faced by engineering students: **the difficulty of finding accurate, up-to-date placement information across multiple companies**. Students often have to:
- Search through multiple PDFs and documents
- Navigate confusing company portals
- Rely on outdated or conflicting information
- Spend hours manually comparing eligibility criteria

### Problem It Solves

This system addresses these challenges by:
- **Centralizing Information**: All placement data in one searchable system
- **Natural Language Queries**: Ask questions in plain English, no technical knowledge needed
- **Accurate Citations**: Every answer includes source references for verification
- **Conflict Detection**: Identifies and resolves conflicting information between official and portal data
- **Real-time Analytics**: Dashboard with system health, pipeline stages, and query history

### What I Learned

Building this project taught me:
- **RAG Architecture**: Deep understanding of retrieval-augmented generation pipelines
- **Hybrid Search**: Combining dense (vector) and sparse (keyword) search for better results
- **Clean Architecture**: Implementing SOLID principles and dependency injection
- **Streamlit UI**: Building interactive, production-ready dashboards
- **Persistent Storage**: Implementing chat history with JSON-based persistence
- **Error Handling**: Graceful fallbacks when models fail (meta tensor errors)

### What Makes This Project Stand Out

- **Multi-Session Chat History**: Persistent chat sessions with auto-generated titles
- **Side-by-Side Dashboard Layout**: Query interface and chat history displayed together
- **Data-Driven Fallback**: Automatic fallback to JSON data when RAG pipeline fails
- **Safety Layer**: Conflict detection and out-of-corpus handling
- **Evaluation System**: 34-query automated evaluation pipeline
- **Production-Ready**: Error handling, logging, and monitoring

---

## Features

- **Hybrid Retrieval System**: Dense vector (FAISS) + Sparse (BM25) with RRF fusion
- **Multimodal Support**: PDF parsing with OCR, table extraction, and vision understanding
- **CrossEncoder Reranking**: Top-20 retrieval → Top-5 reranking for precision
- **Context Refinement**: Overshadow limiter and noise filtering
- **Safety Layer**: Conflict detection, out-of-corpus fallback, confidence scoring
- **Groq LLM Integration**: Fast, cost-effective generation with hallucination reduction
- **Evaluation System**: Automated 34-query evaluation pipeline with metrics
- **Feedback Loop**: AIMD-based continuous improvement
- **ChatGPT-Style UI**: Modern Streamlit interface with chat history and export
- **Multi-Session Chat History**: Persistent chat sessions with auto-generated titles from first query
- **Side-by-Side Dashboard**: Query interface and chat history displayed together for better UX
- **Persistent Storage**: Chat history saved to `data/chat_history.json` - survives refresh and restart
- **Data-Driven Fallback**: Automatic fallback to JSON data when RAG pipeline fails

## Dataset

The system uses a comprehensive placement dataset with 19 companies including:
- **Eligibility Profiles**: CGPA requirements, backlogs allowed, packages, bond periods
- **Interview Experiences**: Round details, technical focus, preparation tips
- **Hiring Distribution**: Role-wise hiring statistics (SDE, Analyst, Officer, Intern)
- **Temporal Trends**: Package trends from 2021-2024
- **Conflicting Records**: Official vs portal-scraped data for conflict detection
- **Overall Statistics**: Aggregate placement statistics

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd placement_rag_sys
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys (Groq, etc.)
```

## Usage

### Streamlit UI (Recommended)

Run the Streamlit interface:

```bash
streamlit run ui/app.py
```

The UI will open in your browser at `http://localhost:8501`

#### Dashboard Features

The dashboard includes:
- **Overview Tab**: System health, pipeline stages, query interface, and chat history
- **Inspection Tab**: Chunk inspector and citation tracer
- **Memory Tab**: Memory analytics
- **Tracing Tab**: Pipeline tracing
- **Settings Tab**: Configuration settings

#### Chat History System

The dashboard features a multi-session chat history system:
- **➕ New Chat Button**: Create new chat sessions
- **🕓 Previous Chat Sessions**: View and switch between previous sessions
- **Auto-Generated Titles**: Session titles are generated from the first user query
- **Persistent Storage**: Chat history is saved to `data/chat_history.json` and survives refresh/restart
- **Side-by-Side Layout**: Query interface and chat history displayed together for better UX

### Python API

```python
from core.pipeline import RAGPipeline
from ingestion.parser import MultimodalParser
from ingestion.chunker import SectionChunker
from ingestion.embedder import SentenceTransformerEmbedder
from retrieval.retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.refiner import ContextRefiner
from generation.prompt_builder import PromptBuilder
from generation.generator import GroqGenerator
from safety.conflict_detector import ConflictDetector

# Initialize components
parser = MultimodalParser()
chunker = SectionChunker()
embedder = SentenceTransformerEmbedder()
vector_store = YourVectorStore()  # Implement IVectorStore
retriever = HybridRetriever(vector_store, embedder)
reranker = CrossEncoderReranker()
refiner = ContextRefiner()
generator = GroqGenerator()
safety_checker = ConflictDetector()

# Create pipeline
pipeline = RAGPipeline(
    parser=parser,
    chunker=chunker,
    embedder=embedder,
    vector_store=vector_store,
    retriever=retriever,
    reranker=reranker,
    refiner=refiner,
    generator=generator,
    safety_checker=safety_checker
)

# Ingest documents
pipeline.ingest("data/Placement_Dataset.pdf")

# Query the system
result = pipeline.query("What is the CGPA requirement for Google?")
print(result["answer"])
```

## Query Examples

### Easy Queries
- "What is the CGPA requirement for TCS?"
- "How many backlogs does Deloitte allow?"
- "What is the package offered by Google?"

### Medium Queries
- "List all companies that allow at least 2 backlogs."
- "Which companies require a CGPA above 8.0?"
- "Which company hires the most Interns?"

### Hard Queries
- "A student with CGPA 7.0, 1 backlog wants maximum pay with no bond."
- "Which Python-focused company hires the most Interns?"
- "Is the Amazon CGPA cutoff 6.4 or 7.0? Explain."

### Expert Queries
- "What is TCS's campus visit date at SVECW?" (Out-of-corpus fallback)
- "I have CGPA 5.0. Where can I apply?" (Below-threshold edge case)

---

## Screenshots

### Dashboard Overview
![Dashboard](screenshots/dashboard.png)

### Chat History System
![Chat History](screenshots/chat_history.png)

### Query Execution
![Query](screenshots/query.png)

*Note: Add screenshots to the `screenshots/` directory to showcase your project.*

---

## Architecture

### 6-Stage RAG Pipeline

1. **Stage 0 - Ingestion**: Parse → Chunk → Embed → Index
   - Multimodal parser (PDF, Excel, OCR)
   - Section-aware chunking with overlap
   - SentenceTransformer embeddings
   - FAISS + BM25 hybrid indexing

2. **Stage 1 - Retrieval**: Dense + Sparse search
   - FAISS vector similarity search
   - BM25 keyword search
   - Reciprocal Rank Fusion (RRF)

3. **Stage 2 - Reranking**: CrossEncoder precision
   - Retrieve top-20 chunks
   - Rerank to top-5 with CrossEncoder
   - Improved contextual precision

4. **Stage 3 - Refinement**: Context optimization
   - ContextRefiner for noise filtering
   - OvershadowLimiter for context cap
   - Deduplication

5. **Stage 4 - Generation**: Grounded prompting
   - PromptBuilder with instructions
   - Groq LLM integration
   - Citation-aware generation

6. **Stage 5 - Safety**: Validation layer
   - Conflict detection (official vs portal)
   - Out-of-corpus fallback
   - Confidence scoring

### Clean Architecture

```
placement_rag_sys/
├── core/                    # SOLID interfaces and orchestrator
│   ├── interfaces.py       # Abstract base classes
│   └── pipeline.py         # 6-stage RAG orchestrator
├── ingestion/              # Stage 0: Parse → Chunk → Embed
│   ├── parser.py          # Multimodal document parser
│   ├── chunker.py         # Section-aware chunking
│   └── embedder.py        # SentenceTransformer embeddings
├── retrieval/              # Stages 1-2: Retrieve → Rerank
│   ├── retriever.py       # Hybrid FAISS + BM25
│   └── reranker.py        # CrossEncoder reranking
├── generation/             # Stages 3-4: Refine → Generate
│   ├── refiner.py         # Context refinement
│   ├── prompt_builder.py  # Grounded prompting
│   └── generator.py       # Groq LLM generation
├── safety/                 # Stage 5: Validation
│   ├── conflict_detector.py    # Conflict detection
│   ├── fallback_guard.py       # Out-of-corpus guard
│   └── overshadow_limiter.py   # Context cap
├── evaluation/             # Evaluation system
│   ├── queries.py         # 34 evaluation queries
│   ├── metrics.py         # Retrieval/answer metrics
│   └── evaluator.py       # Automated evaluator
├── feedback/               # AIMD feedback loop
│   └── loop.py            # Feedback controller
├── config/                 # Configuration management
│   └── settings.py        # System settings
├── ui/                     # Streamlit interface
│   └── app.py            # ChatGPT-style UI
├── scripts/                # Utility scripts
│   ├── ingest.py         # Ingestion pipeline
│   └── evaluate.py       # Evaluation pipeline
└── data/                   # Data storage
    ├── processed/        # Processed data files
    └── raw/             # Raw data files
```

## Configuration

Configuration is managed through `config/settings.py`:
- `EmbeddingConfig`: Model and device settings
- `RetrievalConfig`: Top-k and reranking settings
- `ChunkingConfig`: Chunk size and overlap
- `GenerationConfig`: LLM and context settings
- `SafetyConfig`: Thresholds and flags
- `EvaluationConfig`: Evaluation paths
- `DataConfig`: Data directory paths
- `UIConfig`: UI settings

Environment variables (`.env`):
```
GROQ_API_KEY=your_groq_api_key_here
```

## Evaluation

The system includes 34 evaluation queries:
- **30 Official Queries**: Covering eligibility, interviews, hiring, trends, statistics
- **4 Multi-hop Queries**: Requiring cross-document reasoning

Run evaluation:
```bash
python scripts/evaluate.py
```

Metrics:
- Precision@K, Recall@K, MRR for retrieval
- Exact match, keyword overlap, citation coverage for answers

---

## Testing

### Running Tests

The project includes automated tests for core components:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_pipeline.py

# Run with coverage
pytest --cov=. tests/
```

### Test Structure

```
tests/
├── test_pipeline.py       # RAG pipeline tests
├── test_retrieval.py      # Retrieval system tests
├── test_embeddings.py     # Embedding generation tests
└── test_safety.py         # Safety layer tests
```

### Test Coverage

- **Pipeline Tests**: End-to-end query execution
- **Retrieval Tests**: Dense, sparse, and hybrid search
- **Embedding Tests**: Embedding generation and similarity
- **Safety Tests**: Conflict detection and fallback mechanisms

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new features
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Contribution Guidelines

- Follow the existing code style (PEP 8)
- Add docstrings to new functions and classes
- Update the README if you add new features
- Ensure backward compatibility
- Write tests for new functionality

### Reporting Issues

If you find a bug or have a feature request:
1. Check existing issues to avoid duplicates
2. Use a clear and descriptive title
3. Provide detailed information about the issue
4. Include steps to reproduce the problem
5. Add relevant logs or screenshots if applicable

---

## Scripts

### Ingestion Script

```bash
python scripts/ingest.py --file data/Placement_Dataset.pdf
```

### Evaluation Script

```bash
python scripts/evaluate.py
```

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- **Core**: langchain, streamlit, python-dotenv
- **Embeddings**: sentence-transformers, faiss-cpu
- **Retrieval**: rank-bm25, nltk
- **Reranking**: sentence-transformers (CrossEncoder)
- **LLM**: groq
- **PDF Processing**: docling, pdfplumber, camelot-py, pymupdf
- **Utilities**: numpy, pandas, scikit-learn, tqdm

## Design Principles

- **SOLID Principles**: Abstract interfaces for all components
- **Clean Architecture**: Separation of concerns with clear boundaries
- **Loose Coupling**: Dependency injection and interface-based design
- **Config-Driven**: All settings managed through configuration classes
- **Production-Ready**: Error handling, logging, and monitoring

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: Placement Intelligence Assistant Dataset for RAG-ATHON 24
- Frameworks: LangChain, Streamlit, FAISS
- Models: Sentence-transformers, Groq LLMs
