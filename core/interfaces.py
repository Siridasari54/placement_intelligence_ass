"""Core interfaces and abstract classes following SOLID principles."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document


class IParser(ABC):
    """Abstract interface for document parsers."""
    
    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """Parse document file and return Document objects.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects with extracted content and metadata
        """
        pass


class IChunker(ABC):
    """Abstract interface for document chunking strategies."""
    
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        pass


class IEmbedder(ABC):
    """Abstract interface for embedding generation."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        pass


class IVectorStore(ABC):
    """Abstract interface for vector storage operations."""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
        """
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5, filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of similar document dictionaries
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        pass


class IRetriever(ABC):
    """Abstract interface for document retrieval."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved Document objects
        """
        pass


class IReranker(ABC):
    """Abstract interface for document reranking."""
    
    @abstractmethod
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents based on query relevance.
        
        Args:
            query: Query text
            documents: List of Document objects to rerank
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of Document objects
        """
        pass


class IRefiner(ABC):
    """Abstract interface for context refinement."""
    
    @abstractmethod
    def refine(self, documents: List[Document]) -> List[Document]:
        """Refine and filter documents for better context.
        
        Args:
            documents: List of Document objects to refine
            
        Returns:
            Refined list of Document objects
        """
        pass


class IGenerator(ABC):
    """Abstract interface for answer generation."""
    
    @abstractmethod
    def generate(self, query: str, context: List[Document]) -> str:
        """Generate an answer based on query and context.
        
        Args:
            query: User query
            context: Retrieved context documents
            
        Returns:
            Generated answer string
        """
        pass


class ISafetyChecker(ABC):
    """Abstract interface for safety validation."""
    
    @abstractmethod
    def check_conflict(self, documents: List[Document]) -> List[Document]:
        """Check for conflicts between documents.
        
        Args:
            documents: List of Document objects to check
            
        Returns:
            List of conflicting Document objects
        """
        pass
    
    @abstractmethod
    def check_out_of_corpus(self, query: str, documents: List[Document]) -> bool:
        """Check if query is out of corpus scope.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            True if query is out of corpus, False otherwise
        """
        pass


class IEvaluator(ABC):
    """Abstract interface for evaluation metrics."""
    
    @abstractmethod
    def evaluate_retrieval(self, query: str, retrieved: List[Document], expected: List[str]) -> Dict[str, float]:
        """Evaluate retrieval quality.
        
        Args:
            query: Query text
            retrieved: Retrieved documents
            expected: Expected document IDs
            
        Returns:
            Dictionary of metric scores
        """
        pass
    
    @abstractmethod
    def evaluate_answer(self, query: str, answer: str, expected: str) -> Dict[str, float]:
        """Evaluate answer quality.
        
        Args:
            query: Query text
            answer: Generated answer
            expected: Expected answer
            
        Returns:
            Dictionary of metric scores
        """
        pass


class IFeedbackController(ABC):
    """Abstract interface for feedback loop control."""
    
    @abstractmethod
    def collect_feedback(self, query: str, answer: str, rating: int) -> None:
        """Collect user feedback on answer quality.
        
        Args:
            query: User query
            answer: Generated answer
            rating: User rating (1-5)
        """
        pass
    
    @abstractmethod
    def adjust_parameters(self) -> Dict[str, Any]:
        """Adjust system parameters based on feedback.
        
        Returns:
            Dictionary of adjusted parameters
        """
        pass
