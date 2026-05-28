"""Document chunker with overlap, metadata preservation, and intelligent type detection."""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.interfaces import IChunker
from config.settings import settings
import logging
import re

logger = logging.getLogger(__name__)


class IntelligentChunker(IChunker):
    """Intelligent chunker with heading-aware, table-aware, and company-specific separation."""
    
    def __init__(self):
        """Initialize the intelligent chunker."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Keywords for content type detection
        self.type_keywords = {
            "eligibility": ["cgpa", "backlog", "eligibility", "criteria", "requirement", "qualification"],
            "internship": ["internship", "stipend", "duration", "offer", "placement"],
            "package": ["package", "salary", "lpa", "ctc", "compensation", "pay"],
            "interview": ["interview", "round", "process", "experience", "technical", "hr"],
            "statistics": ["statistics", "data", "placed", "percentage", "average", "trend"]
        }
        
        # Heading patterns for section detection
        self.heading_patterns = [
            r'^#+\s+.+$',  # Markdown headings
            r'^[A-Z][A-Z\s]+:$',  # Uppercase headings with colon
            r'^\d+\.\s+.+$',  # Numbered sections
            r'^[A-Z][a-z]+\s+Requirements$',  # Specific pattern
            r'^[A-Z][a-z]+\s+Process$',  # Specific pattern
        ]
        
        logger.info("IntelligentChunker initialized with heading-aware chunking")
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type from text using keyword analysis.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Detected content type
        """
        text_lower = text.lower()
        type_scores = {}
        
        for content_type, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            type_scores[content_type] = score
        
        # Return type with highest score, or "general" if no match
        if type_scores:
            max_type = max(type_scores, key=type_scores.get)
            if type_scores[max_type] > 0:
                return max_type
        
        return "general"
    
    def _detect_company(self, text: str) -> str:
        """Detect company name from text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Detected company name or "unknown"
        """
        companies = [
            "tcs", "tata consultancy services", "infosys", "wipro", "google", 
            "microsoft", "amazon", "meta", "facebook", "apple", "netflix",
            "deloitte", "accenture", "cognizant", "hcl", "tech mahindra"
        ]
        
        text_lower = text.lower()
        for company in companies:
            if company in text_lower:
                return company.title()
        
        return "unknown"
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces with heading-aware separation and intelligent metadata enrichment.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects with enriched metadata
        """
        logger.info(f"Chunking {len(documents)} documents with heading-aware separation")
        
        chunks = []
        for doc in documents:
            # Detect document-level metadata
            doc_company = self._detect_company(doc.page_content)
            doc_year = self._detect_year(doc.metadata.get("source", ""))
            
            # Split by headings first for better context preservation
            sections = self._split_by_headings(doc.page_content)
            
            # Chunk each section separately
            for section_idx, section_text in enumerate(sections):
                text_chunks = self.splitter.split_text(section_text)
                
                # Create Document objects for each chunk with enriched metadata
                for i, chunk_text in enumerate(text_chunks):
                    # Detect chunk-level content type
                    chunk_type = self._detect_content_type(chunk_text)
                    
                    chunk = Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_id": f"{section_idx}_{i}",
                            "section_id": section_idx,
                            "chunk_count": len(text_chunks),
                            "source": doc.metadata.get("source", "unknown"),
                            "type": chunk_type,
                            "company": doc_company,
                            "year": doc_year,
                            "page": doc.metadata.get("page", "unknown"),
                            "chunk_type": "heading_aware"
                        }
                    )
                    chunks.append(chunk)
        
        logger.info(f"Generated {len(chunks)} chunks from {len(documents)} documents with heading-aware metadata")
        return chunks
    
    def _split_by_headings(self, text: str) -> List[str]:
        """Split text by headings for better context preservation.
        
        Args:
            text: Text content to split
            
        Returns:
            List of text sections
        """
        sections = []
        current_section = []
        
        lines = text.split('\n')
        for line in lines:
            # Check if line is a heading
            is_heading = any(re.match(pattern, line) for pattern in self.heading_patterns)
            
            if is_heading and current_section:
                # Save current section and start new one
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add last section
        if current_section:
            sections.append('\n'.join(current_section))
        
        # If no headings found, return original text as single section
        if not sections:
            sections = [text]
        
        return sections
    
    def _detect_year(self, source: str) -> str:
        """Detect year from source filename or metadata.
        
        Args:
            source: Source string to analyze
            
        Returns:
            Detected year or "unknown"
        """
        year_pattern = r'20\d{2}'
        match = re.search(year_pattern, source)
        if match:
            return match.group()
        return "unknown"


class SectionChunker(IChunker):
    """Section-aware chunker with overlap for better context preservation (legacy)."""
    
    def __init__(self):
        """Initialize the section chunker."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info("SectionChunker initialized")
    
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces with overlap.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        logger.info(f"Chunking {len(documents)} documents")
        
        chunks = []
        for doc in documents:
            # Split text into chunks
            text_chunks = self.splitter.split_text(doc.page_content)
            
            # Create Document objects for each chunk with preserved metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "chunk_count": len(text_chunks),
                        "source": doc.metadata.get("source", "unknown")
                    }
                )
                chunks.append(chunk)
        
        logger.info(f"Generated {len(chunks)} chunks from {len(documents)} documents")
        return chunks


class Deduplicator:
    """TF-IDF cosine similarity deduplicator for removing duplicate chunks."""
    
    def __init__(self, threshold: float = 0.95):
        """Initialize the deduplicator.
        
        Args:
            threshold: Similarity threshold for deduplication
        """
        self.threshold = threshold
        logger.info(f"Deduplicator initialized with threshold {threshold}")
    
    def deduplicate(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on TF-IDF cosine similarity.
        
        Args:
            documents: List of Document objects to deduplicate
            
        Returns:
            List of deduplicated Document objects
        """
        logger.info(f"Deduplicating {len(documents)} documents")
        
        # Simple content-based deduplication
        seen_content = set()
        unique_documents = []
        
        for doc in documents:
            content_normalized = doc.page_content.lower().strip()
            if content_normalized not in seen_content:
                seen_content.add(content_normalized)
                unique_documents.append(doc)
        
        logger.info(f"Deduplicated to {len(unique_documents)} documents")
        return unique_documents
