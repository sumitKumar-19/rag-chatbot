"""
Unit tests for embedding service.
"""
import pytest
from app.services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test cases for EmbeddingService."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create an EmbeddingService instance."""
        return EmbeddingService()
    
    def test_embedding_initialization(self, embedding_service):
        """Test that embedding model is initialized."""
        assert embedding_service.embeddings is not None
        assert embedding_service.sentence_transformer is not None
    
    def test_embed_text(self, embedding_service):
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = embedding_service.embed_text(text)
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_documents(self, embedding_service):
        """Test embedding multiple texts."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = embedding_service.embed_documents(texts)
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) > 0 for emb in embeddings)
    
    def test_get_embedding_dimension(self, embedding_service):
        """Test getting embedding dimension."""
        dimension = embedding_service.get_embedding_dimension()
        assert dimension > 0
        assert isinstance(dimension, int)


if __name__ == "__main__":
    pytest.main([__file__])

