"""
Embedding service using Hugging Face sentence transformers.
Generates vector embeddings for text chunks.
"""
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        """Initialize the embedding model."""
        self.model_name = settings.embedding_model
        self.device = settings.device
        
        # Initialize Hugging Face embeddings
        # Using sentence-transformers for better performance
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
        )
        
        # Also keep direct access to sentence transformer for batch processing
        self.sentence_transformer = SentenceTransformer(
            self.model_name,
            device=self.device
        )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding vector
        """
        embedding = self.embeddings.embed_query(text)
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embeddings.embed_documents(texts)
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        # Test embedding to get dimension
        test_embedding = self.embed_text("test")
        return len(test_embedding)

