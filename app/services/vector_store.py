"""
Vector store service using ChromaDB.
Handles storage and retrieval of document embeddings.
"""
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from app.config import settings
from app.models.document import DocumentChunk
from app.services.embedding_service import EmbeddingService


class VectorStoreService:
    """Service for managing vector store operations."""
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize vector store.
        
        Args:
            embedding_service: Service for generating embeddings
        """
        self.embedding_service = embedding_service
        self.persist_directory = Path(settings.chroma_persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embedding_service.embeddings,
            collection_name="rag_documents"
        )
    
    def add_documents(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Add document chunks to the vector store.
        ChromaDB will automatically generate embeddings using the embedding function.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of document IDs created
        """
        print(f"Processing {len(chunks)} chunks for vector store...")
        
        # Convert DocumentChunks to LangChain Documents
        # ChromaDB will automatically generate embeddings using the embedding_function
        documents = []
        ids = []
        
        for idx, chunk in enumerate(chunks):
            # Create LangChain Document
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "source": chunk.metadata.source,
                    "filename": chunk.metadata.filename,
                    "page_number": chunk.metadata.page_number,
                    "chunk_index": chunk.metadata.chunk_index,
                    "uploaded_at": chunk.metadata.uploaded_at.isoformat() if chunk.metadata.uploaded_at else None
                }
            )
            documents.append(doc)
            
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            # Progress indicator for large documents
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(chunks)} chunks...")
        
        print(f"Adding {len(documents)} documents to vector store (generating embeddings)...")
        
        # Add to vector store - ChromaDB will automatically generate embeddings
        # This is much faster than generating embeddings one by one
        self.vectorstore.add_documents(
            documents=documents,
            ids=ids
        )
        
        print(f"Successfully added {len(ids)} documents to vector store!")
        
        return ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None
    ) -> List[Document]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Search query text
            k: Number of results to return (defaults to config)
            
        Returns:
            List of similar Document objects with metadata
        """
        if k is None:
            k = settings.top_k_retrieval
        
        # Perform similarity search
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k
        )
        
        # Extract documents (results are tuples of (Document, score))
        documents = [doc for doc, score in results]
        
        # Add scores to metadata
        for i, (doc, score) in enumerate(results):
            documents[i].metadata["similarity_score"] = float(score)
        
        return documents
    
    def similarity_search_with_scores(
        self, 
        query: str, 
        k: int = None
    ) -> List[tuple]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of tuples (Document, score)
        """
        if k is None:
            k = settings.top_k_retrieval
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k
        )
        
        return results
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.vectorstore.delete(ids=document_ids)
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        collection = self.vectorstore._collection
        count = collection.count()
        
        return {
            "total_documents": count,
            "persist_directory": str(self.persist_directory),
            "collection_name": "rag_documents"
        }

