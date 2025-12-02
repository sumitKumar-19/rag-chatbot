"""
Unit tests for document service.
"""
import pytest
from pathlib import Path
from app.services.document_service import DocumentService
from app.models.document import DocumentChunk


class TestDocumentService:
    """Test cases for DocumentService."""
    
    @pytest.fixture
    def document_service(self):
        """Create a DocumentService instance."""
        return DocumentService()
    
    def test_text_splitter_initialization(self, document_service):
        """Test that text splitter is initialized correctly."""
        assert document_service.text_splitter is not None
        assert document_service.text_splitter._chunk_size == 1000
        assert document_service.text_splitter._chunk_overlap == 200
    
    def test_upload_directory_creation(self, document_service):
        """Test that upload directory is created."""
        assert document_service.upload_dir.exists()
        assert document_service.upload_dir.is_dir()
    
    def test_chunk_documents(self, document_service):
        """Test document chunking."""
        from langchain.schema import Document
        
        # Create sample documents
        docs = [
            Document(page_content="This is a test document. " * 100, metadata={"page": 1}),
            Document(page_content="Another test document. " * 100, metadata={"page": 2})
        ]
        
        chunks = document_service.chunk_documents(docs, "test.pdf")
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.metadata.filename == "test.pdf" for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__])

