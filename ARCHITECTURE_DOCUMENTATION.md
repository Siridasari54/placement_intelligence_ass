wind# RAG System Architecture Documentation

## Executive Summary

This document describes the refactored architecture of the Placement Intelligence RAG system, transformed from a tightly coupled monolithic design to a modular, engineering-grade architecture following SOLID principles and clean architecture patterns.

## Updated Folder Structure

```
placement_rag_sys/
├── core/
│   ├── interfaces/              # SOLID interfaces
│   │   ├── interfaces.py        # Core interfaces (IParser, IChunker, etc.)
│   │   └── segregated.py        # Segregated interfaces (Interface Segregation Principle)
│   ├── services/                # Service layer with dependency injection
│   │   ├── refactored_rag_service.py  # Refactored RAG service
│   │   └── utility/             # Utility services
│   │       ├── validation_service.py
│   │       └── document_service.py
│   ├── di/                      # Dependency Injection layer
│   │   ├── container.py        # Service container
│   │   └── factories.py        # Service factories
│   ├── observability/           # Logging and metrics
│   │   └── logger.py           # Unified logging infrastructure
│   ├── config/                  # Configuration management
│   │   ├── settings.py         # Settings dataclasses
│   │   └── validators.py       # Configuration validation
│   ├── agents/                  # Agent-based components
│   ├── analytics/               # Retrieval analytics
│   ├── evaluation/              # Advanced evaluation
│   ├── memory/                  # Memory systems
│   ├── multimodal/              # Multimodal intelligence
│   ├── retrieval/              # Adaptive retrieval
│   └── reliability/             # System reliability
├── ingestion/                   # Ingestion implementations
│   ├── parser.py
│   ├── chunker.py
│   └── embedder.py
├── retrieval/                   # Retrieval implementations
│   ├── retriever.py
│   └── reranker.py
├── generation/                  # Generation implementations
│   ├── generator.py
│   ├── refiner.py
│   └── prompt_builder.py
├── safety/                      # Safety implementations
│   ├── conflict_detector.py
│   ├── fallback_guard.py
│   └── overshadow_limiter.py
├── evaluation/                  # Evaluation implementations
│   ├── queries.py
│   ├── metrics.py
│   └── evaluator.py
├── feedback/                    # Feedback implementations
│   └── loop.py
├── ui/                          # User interface
│   ├── app.py                   # ChatGPT-style UI
│   └── dashboard.py             # Analytics dashboard
├── scripts/                     # Utility scripts
│   ├── ingest.py
│   └── evaluate.py
├── config/                      # Configuration
│   └── settings.py
└── data/                        # Data storage
```

## Architecture Explanation

### Layered Architecture

The refactored system follows a clean layered architecture:

1. **Interface Layer** (`core/interfaces/`)
   - Defines contracts for all components
   - Enables loose coupling and testability
   - Applies Interface Segregation Principle

2. **Service Layer** (`core/services/`)
   - Business logic implementation
   - Dependency injection for loose coupling
   - Service-oriented architecture

3. **Dependency Injection Layer** (`core/di/`)
   - Service container for managing dependencies
   - Factory pattern for object creation
   - Singleton and transient lifecycle management

4. **Observability Layer** (`core/observability/`)
   - Unified logging infrastructure
   - Metrics collection
   - Structured logging

5. **Configuration Layer** (`core/config/`)
   - Centralized configuration management
   - Configuration validation
   - Environment-specific settings

6. **Implementation Layer** (`ingestion/`, `retrieval/`, `generation/`, etc.)
   - Concrete implementations of interfaces
   - Pluggable components
   - Easy to extend and modify

### Key Architectural Principles

#### 1. Dependency Inversion Principle (DIP)

**Before:**
```python
class RAGService:
    def __init__(self):
        self.chunking_service = ChunkingService()  # Tight coupling
        self.vector_store = ChromaStore()           # Tight coupling
```

**After:**
```python
class RefactoredRAGService:
    def __init__(self, container: ServiceContainer):
        self.container = container
        # Services injected via container
        self._parser = container.get_service(IParser)
        self._chunker = container.get_service(IChunker)
```

**Benefits:**
- Loose coupling between components
- Easy to swap implementations
- Simplified testing with mocks
- Follows DIP: depends on abstractions, not concretions

#### 2. Interface Segregation Principle (ISP)

**Before:** Large monolithic interfaces
```python
class IParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        pass
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        pass
    
    @abstractmethod
    def extract_tables(self, file_path: str) -> List[List[str]]:
        pass
```

**After:** Segregated focused interfaces
```python
class ITextExtractor(ABC):
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        pass

class ITableExtractor(ABC):
    @abstractmethod
    def extract_tables(self, file_path: str) -> List[List[str]]:
        pass
```

**Benefits:**
- Clients depend only on methods they use
- Smaller, more focused interfaces
- Easier to implement and maintain
- Reduces coupling

#### 3. Open/Closed Principle (OCP)

**Before:** Direct modification for new features
```python
class Retriever:
    def retrieve(self, query: str):
        # Direct implementation
        if self.mode == "vector":
            return self.vector_search(query)
        elif self.mode == "bm25":
            return self.bm25_search(query)
        # Must modify to add new modes
```

**After:** Extension without modification
```python
class IRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        pass

class HybridRetriever(IRetriever):
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        # Hybrid implementation

class AdaptiveRetriever(IRetriever):
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        # Adaptive implementation
```

**Benefits:**
- Open for extension, closed for modification
- New retrievers can be added without changing existing code
- Pluggable architecture

#### 4. Single Responsibility Principle (SRP)

**Before:** Utility-heavy design with mixed responsibilities
```python
# core/utils/helpers.py - mixed responsibilities
def validate_config(config): pass
def format_response(response): pass
def calculate_metrics(data): pass
def normalize_documents(docs): pass
```

**After:** Focused services with single responsibilities
```python
# core/services/utility/validation_service.py
class ValidationService:
    def validate(self, config): pass

# core/services/utility/document_service.py
class DocumentService:
    def normalize_documents(self, docs): pass

# core/observability/logger.py
class MetricsCollector:
    def calculate_metrics(self, data): pass
```

**Benefits:**
- Each class has one reason to change
- Easier to understand and maintain
- Reduced coupling
- Better testability

## Design Pattern Usage

### 1. Dependency Injection Pattern

**Implementation:** `core/di/container.py`

```python
class ServiceContainer(IServiceProvider):
    def get_service(self, service_type: Type[T]) -> T:
        # Resolve dependencies automatically
        if service_type in self._singletons:
            return self._singletons[service_type]
        if service_type in self._factories:
            factory = self._factories[service_type]
            dependencies = self._resolve_dependencies(factory)
            return factory(**dependencies)
```

**Benefits:**
- Loose coupling
- Testability with mocks
- Centralized dependency management
- Lifecycle management (singleton/transient)

### 2. Factory Pattern

**Implementation:** `core/di/factories.py`

```python
class ServiceFactory:
    def create_retriever(self) -> IRetriever:
        vector_store = self.container.get_service(IVectorStore)
        embedder = self.container.get_service(IEmbedder)
        return HybridRetriever(vector_store=vector_store, embedder=embedder)
```

**Benefits:**
- Encapsulates complex object creation
- Centralizes instantiation logic
- Enables dependency resolution
- Easy to extend with new factories

### 3. Strategy Pattern

**Implementation:** `core/retrieval/adaptive_strategy.py`

```python
class AdaptiveRetrievalStrategy:
    def select_mode(self, query: str) -> RetrievalMode:
        # Dynamically select strategy based on query characteristics
        if self._is_semantic_query(query):
            return RetrievalMode.SEMANTIC_HEAVY
        elif self._is_keyword_query(query):
            return RetrievalMode.KEYWORD_HEAVY
```

**Benefits:**
- Pluggable algorithms
- Runtime strategy selection
- Easy to add new strategies
- Follows Open/Closed Principle

### 4. Observer Pattern

**Implementation:** `core/observability/logger.py`

```python
class MetricsCollector:
    def record_latency(self, latency_ms: float) -> None:
        self.metrics["total_latency_ms"] += latency_ms
        # Observers can subscribe to metric changes
```

**Benefits:**
- Event-driven architecture
- Decoupled notification system
- Real-time monitoring
- Extensible observability

### 5. Repository Pattern

**Implementation:** `core/interfaces.py` - IVectorStore

```python
class IVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        pass
```

**Benefits:**
- Data access abstraction
- Swappable storage backends
- Testable with in-memory implementations
- Clean separation of concerns

## Low Coupling / High Cohesion Improvements

### Before Refactoring

**High Coupling Issues:**
- Direct imports between services (`from app.vectorstore.chroma_store import ChromaStore`)
- Hard-coded dependencies in constructors
- Mixed responsibilities in utility modules
- Tight coupling to specific implementations

**Low Cohesion Issues:**
- Utility modules with unrelated functions
- Services doing multiple things
- Configuration scattered across files
- No clear separation of concerns

### After Refactoring

**Low Coupling Achieved:**
- All dependencies injected via interfaces
- Service container manages dependencies
- No direct imports between service modules
- Components depend on abstractions, not concretions

**High Cohesion Achieved:**
- Each service has single responsibility
- Related functionality grouped together
- Clear module boundaries
- Focused, purpose-built components

### Example: RAG Service Refactoring

**Before (High Coupling):**
```python
class RAGService:
    def __init__(self):
        self.chunking_service = ChunkingService()  # Direct instantiation
        self.vector_store = ChromaStore(persist_dir="...")  # Direct instantiation
        self.retrieval_service = RetrievalService()  # Direct instantiation
```

**After (Low Coupling):**
```python
class RefactoredRAGService:
    def __init__(self, container: ServiceContainer):
        self.container = container
        # Dependencies injected via container
        self._parser = container.get_service(IParser)
        self._chunker = container.get_service(IChunker)
        self._embedder = container.get_service(IEmbedder)
```

## Reusable Interfaces

### Core Interfaces (`core/interfaces/interfaces.py`)

1. **IParser** - Document parsing abstraction
2. **IChunker** - Document chunking abstraction
3. **IEmbedder** - Embedding generation abstraction
4. **IVectorStore** - Vector storage abstraction
5. **IRetriever** - Document retrieval abstraction
6. **IReranker** - Document reranking abstraction
7. **IRefiner** - Context refinement abstraction
8. **IGenerator** - Answer generation abstraction
9. **ISafetyChecker** - Safety validation abstraction
10. **IEvaluator** - Evaluation metrics abstraction
11. **IFeedbackController** - Feedback control abstraction

### Segregated Interfaces (`core/interfaces/segregated.py`)

1. **ITextExtractor** - Text extraction only
2. **ITableExtractor** - Table extraction only
3. **IImageProcessor** - Image processing only
4. **IQueryRewriter** - Query rewriting only
5. **IQueryClassifier** - Query classification only
6. **IMetadataFilter** - Metadata filtering only
7. **IPromptBuilder** - Prompt building only
8. **IResponseFormatter** - Response formatting only
9. **IFaithfulnessChecker** - Faithfulness checking only
10. **IHallucinationDetector** - Hallucination detection only
11. **ICitationValidator** - Citation validation only
12. **ICache** - Caching abstraction
13. **IConversationStore** - Conversation storage abstraction
14. **IConfigurationProvider** - Configuration provision abstraction
15. **IConfigurationValidator** - Configuration validation abstraction

## Engineering-Level Improvements

### 1. Observability

**Implementation:** `core/observability/logger.py`

**Features:**
- Structured logging with JSON format
- Metrics collection (latency, cache hits, errors)
- File and console handlers
- Contextual logging with extra fields

**Benefits:**
- Production-ready monitoring
- Easy debugging
- Performance tracking
- Error analysis

### 2. Configuration Management

**Implementation:** `core/config/validators.py`

**Features:**
- Centralized configuration
- Validation rules per section
- Type and range validation
- Default values
- Dot notation access

**Benefits:**
- Single source of truth
- Type safety
- Validation at startup
- Easy to maintain

### 3. Service-Based Architecture

**Implementation:** `core/services/`

**Features:**
- Focused services with single responsibilities
- Dependency injection
- Interface-based design
- Easy to test and mock

**Benefits:**
- Maintainability
- Testability
- Scalability
- Reusability

### 4. Error Handling

**Implementation:** Throughout refactored code

**Features:**
- Consistent error handling
- Proper exception propagation
- Logging of errors with context
- Graceful degradation

**Benefits:**
- Production reliability
- Debugging support
- User experience
- System stability

### 5. Type Safety

**Implementation:** Throughout refactored code

**Features:**
- Type hints on all functions
- Interface typing
- Generic types where appropriate
- Type checking with mypy

**Benefits:**
- Catch errors at development time
- Better IDE support
- Self-documenting code
- Refactoring safety

## Migration Guide

### Step 1: Initialize Service Container

```python
from core.di.container import ServiceContainer
from core.di.factories import register_services

# Create container
container = ServiceContainer()

# Register services
register_services(container)

# Set configuration
container.set_config(your_config_dict)
```

### Step 2: Use Refactored Service

```python
from core.services.refactored_rag_service import RefactoredRAGService

# Create service with dependency injection
rag_service = RefactoredRAGService(container)

# Initialize
rag_service.initialize()

# Query
result = rag_service.query("What is the CGPA requirement for Google?")
```

### Step 3: Use New Interfaces

```python
from core.interfaces.segregated import ITextExtractor, ITableExtractor

# Inject implementations
text_extractor = container.get_service(ITextExtractor)
table_extractor = container.get_service(ITableExtractor)

# Use interfaces
text = text_extractor.extract_text("document.pdf")
tables = table_extractor.extract_tables("document.pdf")
```

## Testing Strategy

### Unit Testing

```python
# Mock dependencies using interfaces
from unittest.mock import Mock
from core.interfaces import IRetriever

# Create mock
mock_retriever = Mock(spec=IRetriever)
mock_retriever.retrieve.return_value = [Document(page_content="test")]

# Inject into service
service = RefactoredRAGService(container)
service._retriever = mock_retriever

# Test
result = service.query("test query")
assert result["status"] == "success"
```

### Integration Testing

```python
# Use real container with test configuration
container = ServiceContainer()
container.set_config(test_config)
register_services(container)

# Test full flow
service = RefactoredRAGService(container)
service.initialize()
result = service.query("test query")
```

## Performance Considerations

### Dependency Injection Overhead

- **Impact:** Minimal (< 1ms per resolution)
- **Mitigation:** Singleton caching for frequently used services
- **Trade-off:** Worth it for maintainability and testability

### Service Resolution

- **Strategy:** Lazy loading of services
- **Benefit:** Faster startup
- **Trade-off:** Slightly slower first access

### Configuration Validation

- **Impact:** One-time cost at startup
- **Benefit:** Early error detection
- **Trade-off:** Worth it for reliability

## Conclusion

The refactored RAG system now follows enterprise-grade architecture principles:

- **SOLID Principles:** Applied throughout
- **Clean Architecture:** Clear layer separation
- **Dependency Injection:** Loose coupling
- **Interface Segregation:** Focused interfaces
- **Observability:** Production-ready logging and metrics
- **Configuration Management:** Centralized and validated
- **Service-Based Architecture:** Maintainable and scalable

The system is now:
- **Easier to maintain** - Clear structure and single responsibilities
- **Easier to test** - Dependency injection and interfaces
- **Easier to extend** - Open/Closed Principle and pluggable components
- **Easier to understand** - Clear architecture and documentation
- **Production-ready** - Observability, error handling, and validation

This refactoring transforms the codebase from a tightly coupled monolithic design to a modular, engineering-grade architecture suitable for production use and long-term maintenance.
