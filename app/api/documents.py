"""
API endpoints for document management.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
import uuid
from datetime import datetime

from app.models.document import DocumentUploadResponse, DocumentInfo
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService


router = APIRouter(prefix="/api/documents", tags=["documents"])


# Dependency injection for services
def get_document_service() -> DocumentService:
    """Get document service instance."""
    return DocumentService()


def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    return EmbeddingService()


def get_vector_store_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service)
) -> VectorStoreService:
    """Get vector store service instance."""
    return VectorStoreService(embedding_service)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Upload and process a PDF document.
    
    - **file**: PDF file to upload
    - Returns document ID and processing status
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Save file
        file_path = document_service.save_uploaded_file(
            file_content,
            file.filename
        )
        
        # Process document (load and chunk)
        chunks = document_service.process_uploaded_file(
            file_path,
            file.filename
        )
        
        # Add to vector store
        document_ids = vector_store.add_documents(chunks)
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            chunks_created=len(chunks),
            message=f"Successfully processed {len(chunks)} chunks from {file.filename}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@router.get("/info", response_model=dict)
async def get_vector_store_info(
    vector_store: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Get information about the vector store.
    
    Returns statistics about stored documents.
    """
    try:
        info = vector_store.get_collection_info()
        return info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving vector store info: {str(e)}"
        )


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    vector_store: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Delete a document from the vector store.
    
    - **document_id**: ID of the document to delete
    """
    try:
        success = vector_store.delete_documents([document_id])
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

