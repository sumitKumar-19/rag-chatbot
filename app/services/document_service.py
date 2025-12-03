"""
Document ingestion service.
Handles loading, chunking, and preparing documents for vector storage.
"""
import os
from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import settings
from app.models.document import DocumentChunk, DocumentMetadata
from datetime import datetime


class DocumentService:
    """Service for processing and chunking documents."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.upload_dir = Path("./data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and extract text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with page content
        """
        try:
            print(f"Loading PDF from: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from PDF")
            return documents
        except Exception as e:
            raise ValueError(f"Error loading PDF: {str(e)}")
    
    def chunk_documents(
        self, 
        documents: List[Document], 
        filename: str
    ) -> List[DocumentChunk]:
        """
        Split documents into chunks with metadata.
        
        Args:
            documents: List of LangChain Document objects
            filename: Original filename
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Split documents into chunks
        print("Splitting documents into chunks...")
        text_chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(text_chunks)} chunks")
        
        for idx, chunk in enumerate(text_chunks):
            # Extract page number from metadata if available
            page_number = chunk.metadata.get("page", None)
            
            metadata = DocumentMetadata(
                source=filename,
                filename=filename,
                page_number=page_number,
                chunk_index=idx,
                uploaded_at=datetime.now()
            )
            
            doc_chunk = DocumentChunk(
                content=chunk.page_content,
                metadata=metadata
            )
            
            chunks.append(doc_chunk)
        
        return chunks
    
    def process_uploaded_file(
        self, 
        file_path: str, 
        filename: str
    ) -> List[DocumentChunk]:
        """
        Complete pipeline: load and chunk a document.
        
        Args:
            file_path: Path to uploaded file
            filename: Original filename
            
        Returns:
            List of DocumentChunk objects ready for embedding
        """
        # Load document
        documents = self.load_pdf(file_path)
        
        # Chunk documents
        chunks = self.chunk_documents(documents, filename)
        
        return chunks
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> str:
        """
        Save uploaded file to disk.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        file_path = self.upload_dir / filename
        with open(file_path, "wb") as f:
            f.write(file_content)
        return str(file_path)

