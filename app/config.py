"""
Configuration management for the RAG Chatbot application.
Uses Pydantic Settings for type-safe configuration with environment variables.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Hugging Face Configuration
    huggingface_api_token: Optional[str] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "microsoft/DialoGPT-medium"
    use_local_models: bool = True
    device: str = "cpu"  # 'cpu' or 'cuda'
    
    # Application Configuration
    app_name: str = "RAG Chatbot"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # Vector Database Configuration
    vector_db_type: str = "chroma"
    chroma_persist_directory: str = "./data/vectorstore"
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 5
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

