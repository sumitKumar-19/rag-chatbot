"""
API endpoints for chat and query operations.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from app.models.query import QueryRequest, QueryResponse
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.rag_service import RAGService


router = APIRouter(prefix="/api/chat", tags=["chat"])


# Dependency injection
def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    return EmbeddingService()


def get_vector_store_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> VectorStoreService:
    """Get vector store service instance."""
    return VectorStoreService(embedding_service)


def get_rag_service(
    vector_store: VectorStoreService = Depends(get_vector_store_service)
) -> RAGService:
    """Get RAG service instance."""
    return RAGService(vector_store)


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Process a RAG query.
    
    - **query**: User's question
    - **top_k**: Number of documents to retrieve (optional)
    - **temperature**: LLM temperature (optional)
    - **max_tokens**: Maximum tokens in response (optional)
    
    Returns answer with source citations.
    """
    try:
        response = rag_service.query(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/simple-query", response_model=QueryResponse)
async def simple_query(
    query: str,
    top_k: Optional[int] = None,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Simplified query endpoint.
    
    - **query**: User's question
    - **top_k**: Number of documents to retrieve (optional)
    """
    try:
        response = rag_service.simple_query(query, top_k)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

