"""Multi-hop retrieval system with query rewriting and iterative refinement."""

from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from core.interfaces import IRetriever
import logging

logger = logging.getLogger(__name__)


class MultiHopRetriever:
    """Multi-hop retriever with iterative query refinement and evidence merging."""
    
    def __init__(
        self,
        base_retriever: IRetriever,
        max_hops: int = 3,
        min_confidence: float = 0.6,
        enable_query_rewriting: bool = True
    ):
        """Initialize the multi-hop retriever.
        
        Args:
            base_retriever: Base retriever to use for retrieval
            max_hops: Maximum number of retrieval hops
            min_confidence: Minimum confidence threshold to stop hopping
            enable_query_rewriting: Whether to enable query rewriting
        """
        self.base_retriever = base_retriever
        self.max_hops = max_hops
        self.min_confidence = min_confidence
        self.enable_query_rewriting = enable_query_rewriting
        logger.info(f"MultiHopRetriever initialized with max_hops={max_hops}")
    
    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """Perform multi-hop retrieval with iterative refinement.
        
        Args:
            query: Original query
            k: Number of documents to retrieve per hop
            
        Returns:
            Merged list of retrieved documents from all hops
        """
        logger.info(f"Starting multi-hop retrieval for query: {query}")
        
        all_documents = []
        current_query = query
        hop_count = 0
        
        while hop_count < self.max_hops:
            logger.info(f"Hop {hop_count + 1}: Retrieving with query: {current_query}")
            
            # Retrieve documents for current query
            documents = self.base_retriever.retrieve(current_query, k=k)
            
            if not documents:
                logger.warning(f"Hop {hop_count + 1}: No documents retrieved, stopping")
                break
            
            # Calculate confidence of retrieved documents
            confidence = self._calculate_retrieval_confidence(documents)
            logger.info(f"Hop {hop_count + 1}: Retrieved {len(documents)} documents, confidence: {confidence:.2f}")
            
            # Add documents to collection
            all_documents.extend(documents)
            
            # Check if we have sufficient confidence
            if confidence >= self.min_confidence:
                logger.info(f"Confidence threshold met, stopping at hop {hop_count + 1}")
                break
            
            # Detect missing context and rewrite query
            if self.enable_query_rewriting and hop_count < self.max_hops - 1:
                rewritten_query = self._rewrite_query(current_query, documents)
                if rewritten_query != current_query:
                    logger.info(f"Rewrote query: {current_query} -> {rewritten_query}")
                    current_query = rewritten_query
                else:
                    logger.info("Query rewrite did not change query, stopping")
                    break
            else:
                break
            
            hop_count += 1
        
        # Deduplicate and return merged documents
        merged_documents = self._merge_and_deduplicate(all_documents)
        logger.info(f"Multi-hop retrieval completed: {len(merged_documents)} unique documents")
        
        return merged_documents
    
    def _calculate_retrieval_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score for retrieved documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Confidence score between 0 and 1
        """
        if not documents:
            return 0.0
        
        # Use rerank scores if available, otherwise default to 0.5
        scores = []
        for doc in documents:
            if "rerank_score" in doc.metadata:
                scores.append(doc.metadata["rerank_score"])
            elif "confidence" in doc.metadata:
                scores.append(doc.metadata["confidence"])
            else:
                scores.append(0.5)
        
        if not scores:
            return 0.5
        
        # Return average score
        return sum(scores) / len(scores)
    
    def _rewrite_query(self, query: str, documents: List[Document]) -> str:
        """Rewrite query based on retrieved documents to fill missing context.
        
        Args:
            query: Current query
            documents: Retrieved documents
            
        Returns:
            Rewritten query
        """
        # Extract key entities and context from documents
        entities = self._extract_entities(documents)
        context_keywords = self._extract_context_keywords(documents)
        
        # If we found specific entities, add them to the query
        if entities:
            entity_string = " ".join(entities[:3])  # Limit to top 3 entities
            rewritten = f"{query} {entity_string}"
            return rewritten
        
        # If we found context keywords, add them to the query
        if context_keywords:
            context_string = " ".join(context_keywords[:2])  # Limit to top 2 keywords
            rewritten = f"{query} {context_string}"
            return rewritten
        
        # No rewrite needed
        return query
    
    def _extract_entities(self, documents: List[Document]) -> List[str]:
        """Extract key entities (companies, technologies, etc.) from documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Common company names
        companies = [
            "tcs", "tata consultancy services", "infosys", "wipro", "google",
            "microsoft", "amazon", "meta", "facebook", "apple", "netflix",
            "deloitte", "accenture", "cognizant", "hcl", "tech mahindra"
        ]
        
        # Common technologies
        technologies = [
            "python", "java", "javascript", "react", "angular", "nodejs",
            "machine learning", "ai", "data science", "cloud", "aws", "azure"
        ]
        
        all_entities = companies + technologies
        
        for doc in documents:
            text_lower = doc.page_content.lower()
            for entity in all_entities:
                if entity in text_lower and entity not in entities:
                    entities.append(entity)
        
        return entities
    
    def _extract_context_keywords(self, documents: List[Document]) -> List[str]:
        """Extract context keywords from documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            List of context keywords
        """
        keywords = []
        
        # Common placement-related keywords
        placement_keywords = [
            "eligibility", "criteria", "package", "salary", "cgpa", "backlog",
            "internship", "stipend", "interview", "process", "round", "technical",
            "placement", "offer", "selection", "shortlist", "campus"
        ]
        
        for doc in documents:
            text_lower = doc.page_content.lower()
            for keyword in placement_keywords:
                if keyword in text_lower and keyword not in keywords:
                    keywords.append(keyword)
        
        return keywords
    
    def _merge_and_deduplicate(self, documents: List[Document]) -> List[Document]:
        """Merge documents from multiple hops and remove duplicates.
        
        Args:
            documents: List of documents from all hops
            
        Returns:
            Deduplicated list of documents
        """
        seen_content = set()
        unique_documents = []
        
        for doc in documents:
            # Normalize content for comparison
            content_normalized = doc.page_content.lower().strip()
            
            if content_normalized not in seen_content:
                seen_content.add(content_normalized)
                unique_documents.append(doc)
        
        return unique_documents
