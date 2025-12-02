"""
RAG (Retrieval-Augmented Generation) service.
Orchestrates the complete RAG pipeline: retrieval + generation.
"""
from typing import List, Optional
from datetime import datetime
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from app.config import settings
from app.models.query import QueryRequest, QueryResponse, SourceCitation
from app.services.vector_store import VectorStoreService


class RAGService:
    """Service for RAG query processing."""
    
    def __init__(self, vector_store: VectorStoreService):
        """
        Initialize RAG service.
        
        Args:
            vector_store: Vector store service for retrieval
        """
        self.vector_store = vector_store
        self.llm = self._initialize_llm()
        # Removed unused qa_chain initialization
    
    def _initialize_llm(self):
        """
        Initialize the Hugging Face LLM.
        Uses a simple pipeline approach for CPU-friendly models.
        
        Returns:
            Initialized LLM pipeline
        """
        model_name = settings.llm_model
        device = settings.device
        
        try:
            # Use HuggingFacePipeline with model name directly
            # This is simpler and works well for smaller models
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                model_kwargs={
                    "temperature": settings.temperature,
                    "max_length": settings.max_tokens + 50,  # Add buffer
                    "device_map": "auto" if device == "cuda" else None,
                },
                pipeline_kwargs={
                    "max_new_tokens": settings.max_tokens,
                    "temperature": settings.temperature,
                }
            )
            return llm
            
        except Exception as e:
            print(f"Error initializing LLM with {model_name}: {e}")
            print("Falling back to GPT-2 (lightweight model)...")
            # Fallback to a very lightweight model
            return self._initialize_fallback_llm()
    
    def _initialize_fallback_llm(self):
        """Initialize a lightweight fallback LLM (GPT-2)."""
        try:
            # Use GPT-2 as fallback - very small, always available
            llm = HuggingFacePipeline.from_model_id(
                model_id="gpt2",
                task="text-generation",
                model_kwargs={
                    "temperature": settings.temperature,
                    "max_length": settings.max_tokens + 50,
                },
                pipeline_kwargs={
                    "max_new_tokens": settings.max_tokens,
                    "temperature": settings.temperature,
                }
            )
            return llm
        except Exception as e:
            raise RuntimeError(f"Failed to initialize fallback LLM: {e}")
    
    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a RAG query.
        
        Args:
            request: Query request with question and parameters
            
        Returns:
            Query response with answer and sources
        """
        # Retrieve relevant documents
        k = request.top_k or settings.top_k_retrieval
        retrieved_docs = self.vector_store.similarity_search_with_scores(
            query=request.query,
            k=k
        )
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page_number', 'N/A')}]\n{doc.page_content}"
            for doc, score in retrieved_docs
        ])
        
        # Generate answer using LLM
        # For simpler models, we'll use a direct approach
        prompt = f"""Based on the following context, answer the question. 
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {request.query}

Answer:"""
        
        try:
            answer = self.llm(prompt)
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = "I apologize, but I encountered an error while generating a response."
        
        # Format sources
        sources = [
            SourceCitation(
                content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                source=doc.metadata.get("source", "Unknown"),
                page_number=doc.metadata.get("page_number"),
                score=float(score)
            )
            for doc, score in retrieved_docs
        ]
        
        return QueryResponse(
            answer=answer.strip(),
            sources=sources,
            query=request.query,
            timestamp=datetime.now()
        )
    
    def simple_query(self, query: str, top_k: Optional[int] = None) -> QueryResponse:
        """
        Simplified query interface.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Query response
        """
        request = QueryRequest(query=query, top_k=top_k)
        return self.query(request)