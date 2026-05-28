"""Ingestion script for Placement Intelligence PDF dataset.

This script implements the recommended chunking strategy from the RAG-ATHON 24 dataset:
- Table extraction using pdfplumber/camelot (not PyPDFLoader)
- Deduplication for interview experiences (3x repetition)
- Row-per-company chunking for eligibility tables
- Semantic split for interview text (200-300 tokens)
- Metadata tagging (company, section, year, conflict flag)
- Target: 80-120 meaningful chunks (not 400+ unoptimized)
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vectorstore.vectorstore_factory import VectorStoreFactory
from ingestion.embedder import SentenceTransformerEmbedder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlacementPDFIngestor:
    """Specialized ingestor for Placement Intelligence PDF dataset."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.embedder = SentenceTransformerEmbedder()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.chunks = []
        self.dedup_cache = set()
        
    def extract_tables_from_pdf(self) -> List[pd.DataFrame]:
        """Extract tables from PDF using pdfplumber."""
        tables = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:  # At least header + 1 row
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df['page_num'] = page_num
                        tables.append(df)
        return tables
    
    def extract_text_from_pdf(self) -> str:
        """Extract text from PDF."""
        text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
        return text
    
    def deduplicate_text(self, text: str) -> bool:
        """Check if text is duplicate using hash."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.dedup_cache:
            return True
        self.dedup_cache.add(text_hash)
        return False
    
    def process_eligibility_table(self, df: pd.DataFrame) -> List[Document]:
        """Process eligibility table - row-per-company chunking."""
        documents = []
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Skip if not eligibility table (check for key columns)
        required_cols = ['Company', 'Min CGPA', 'Package']
        if not any(col in df.columns for col in required_cols):
            return documents
        
        for idx, row in df.iterrows():
            company = row.get('Company', '').strip()
            if not company or company == 'Company':
                continue
            
            # Create natural language description
            text_parts = []
            text_parts.append(f"{company} Eligibility Criteria:")
            
            for col in df.columns:
                if col not in ['Company', 'page_num']:
                    value = row.get(col, '')
                    if pd.notna(value) and str(value).strip():
                        text_parts.append(f"{col}: {value}")
            
            text = " | ".join(text_parts)
            
            doc = Document(
                page_content=text,
                metadata={
                    'company': company,
                    'section': 'eligibility',
                    'source': 'placement_dataset.pdf',
                    'chunk_type': 'table_row'
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_interview_text(self, text: str) -> List[Document]:
        """Process interview experiences with deduplication and semantic chunking."""
        documents = []
        
        # Split by company sections
        sections = text.split('|')
        
        for section in sections:
            if not section.strip():
                continue
            
            # Extract company name
            lines = section.strip().split('\n')
            company_line = lines[0] if lines else ""
            company = company_line.split('|')[0].strip() if '|' in company_line else company_line.strip()
            
            if not company or len(company) < 2:
                continue
            
            # Check for deduplication
            if self.deduplicate_text(section.strip()):
                logger.info(f"Skipping duplicate interview content for {company}")
                continue
            
            # Semantic chunking
            chunks = self.text_splitter.split_text(section.strip())
            
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'company': company,
                        'section': 'interview',
                        'chunk_index': idx,
                        'source': 'placement_dataset.pdf',
                        'chunk_type': 'interview_text'
                    }
                )
                documents.append(doc)
        
        return documents
    
    def process_hiring_data(self, df: pd.DataFrame) -> List[Document]:
        """Process hiring distribution data - row-per-company chunking."""
        documents = []
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Skip if not hiring table
        if 'Company' not in df.columns:
            return documents
        
        for idx, row in df.iterrows():
            company = row.get('Company', '').strip()
            if not company or company == 'Company':
                continue
            
            # Create natural language description
            text_parts = []
            text_parts.append(f"{company} Hiring Distribution:")
            
            for col in df.columns:
                if col not in ['Company', 'page_num', 'Total']:
                    value = row.get(col, '')
                    if pd.notna(value) and str(value).strip():
                        text_parts.append(f"{col}: {value}")
            
            text = " | ".join(text_parts)
            
            doc = Document(
                page_content=text,
                metadata={
                    'company': company,
                    'section': 'hiring',
                    'source': 'placement_dataset.pdf',
                    'chunk_type': 'hiring_row'
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_temporal_data(self, df: pd.DataFrame) -> List[Document]:
        """Process temporal trend data - row-per-company per year chunking."""
        documents = []
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Skip if not temporal table
        if 'Company' not in df.columns:
            return documents
        
        for idx, row in df.iterrows():
            company = row.get('Company', '').strip()
            if not company or company == 'Company':
                continue
            
            # Extract year columns (2021, 2022, 2023, 2024)
            year_cols = [col for col in df.columns if any(year in str(col) for year in ['2021', '2022', '2023', '2024'])]
            
            for year_col in year_cols:
                year = year_col.strip()
                value = row.get(year_col, '')
                
                if pd.notna(value) and str(value).strip():
                    text = f"{company} Package in {year}: {value} LPA"
                    
                    doc = Document(
                        page_content=text,
                        metadata={
                            'company': company,
                            'section': 'temporal',
                            'year': year,
                            'source': 'placement_dataset.pdf',
                            'chunk_type': 'temporal_year'
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    def process_conflicting_data(self, df: pd.DataFrame) -> List[Document]:
        """Process conflicting data - keep both with conflict flag."""
        documents = []
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Skip if not conflict table
        if 'Company' not in df.columns or 'CGPA (Official)' not in df.columns:
            return documents
        
        for idx, row in df.iterrows():
            company = row.get('Company', '').strip()
            if not company or company == 'Company':
                continue
            
            # Official record
            official_cgpa = row.get('CGPA (Official)', '')
            official_package = row.get('Package Official', '')
            
            if pd.notna(official_cgpa) and str(official_cgpa).strip():
                text = f"{company} Official Criteria - CGPA: {official_cgpa}, Package: {official_package}"
                
                doc = Document(
                    page_content=text,
                    metadata={
                        'company': company,
                        'section': 'conflict',
                        'source': 'official',
                        'conflict': True,
                        'source': 'placement_dataset.pdf',
                        'chunk_type': 'conflict_record'
                    }
                )
                documents.append(doc)
            
            # Portal record
            portal_cgpa = row.get('CGPA (Portal)', '')
            portal_package = row.get('Package Portal', '')
            
            if pd.notna(portal_cgpa) and str(portal_cgpa).strip():
                text = f"{company} Portal Criteria - CGPA: {portal_cgpa}, Package: {portal_package}"
                
                doc = Document(
                    page_content=text,
                    metadata={
                        'company': company,
                        'section': 'conflict',
                        'source': 'portal',
                        'conflict': True,
                        'source': 'placement_dataset.pdf',
                        'chunk_type': 'conflict_record'
                    }
                )
                documents.append(doc)
        
        return documents
    
    def ingest(self) -> int:
        """Main ingestion pipeline."""
        logger.info(f"Starting ingestion of {self.pdf_path}")
        
        # Extract tables and text
        tables = self.extract_tables_from_pdf()
        text = self.extract_text_from_pdf()
        
        logger.info(f"Extracted {len(tables)} tables from PDF")
        
        # Process each table based on content type
        all_documents = []
        
        for table in tables:
            # Heuristic to determine table type
            if 'Company' in table.columns:
                if 'CGPA (Official)' in table.columns:
                    # Conflicting data
                    docs = self.process_conflicting_data(table)
                    all_documents.extend(docs)
                elif any(year in str(col) for col in table.columns for year in ['2021', '2022', '2023', '2024']):
                    # Temporal data
                    docs = self.process_temporal_data(table)
                    all_documents.extend(docs)
                elif 'SDE' in table.columns or 'Analyst' in table.columns:
                    # Hiring data
                    docs = self.process_hiring_data(table)
                    all_documents.extend(docs)
                elif 'Min CGPA' in table.columns or 'Package' in table.columns:
                    # Eligibility data
                    docs = self.process_eligibility_table(table)
                    all_documents.extend(docs)
        
        # Process interview text
        interview_docs = self.process_interview_text(text)
        all_documents.extend(interview_docs)
        
        logger.info(f"Total documents created: {len(all_documents)}")
        logger.info(f"Deduplication cache size: {len(self.dedup_cache)}")
        
        # Add to vector store
        if all_documents:
            self.add_to_vector_store(all_documents)
        
        return len(all_documents)
    
    def add_to_vector_store(self, documents: List[Document]):
        """Add documents to vector store."""
        logger.info("Adding documents to vector store...")
        
        # Convert to dict format
        docs_to_add = []
        for doc in documents:
            docs_to_add.append({
                'text': doc.page_content,
                'metadata': doc.metadata
            })
        
        # Get vector store
        vector_store = VectorStoreFactory.get_store("faiss")
        
        # Add documents
        vector_store.add_documents(docs_to_add)
        
        logger.info(f"Successfully added {len(docs_to_add)} documents to vector store")


def main():
    """Main entry point."""
    pdf_path = "data/raw/placement_dataset.pdf"
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found at {pdf_path}")
        return
    
    ingestor = PlacementPDFIngestor(pdf_path)
    num_docs = ingestor.ingest()
    
    logger.info(f"Ingestion complete. Created {num_docs} chunks.")
    logger.info("Target: 80-120 meaningful chunks for optimal RAG performance.")


if __name__ == "__main__":
    main()
