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
import threading
import time

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
        Uses a direct pipeline approach for CPU-friendly models.
        
        Returns:
            Initialized LLM pipeline
        """
        model_name = settings.llm_model
        device = settings.device
        
        try:
            # Use direct transformers pipeline approach (more reliable)
            from transformers import pipeline as hf_pipeline
            
            print(f"Loading LLM model: {model_name}...")
            
            # Create pipeline directly
            # For GPT-2, use very conservative token limits
            max_new_tokens = 50 if "gpt2" in model_name.lower() else min(settings.max_tokens, 100)
            pipe = hf_pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_new_tokens=max_new_tokens,
                temperature=settings.temperature,
                device=-1 if device == "cpu" else 0,
                return_full_text=False,
                do_sample=True,
                pad_token_id=50256  # GPT-2 pad token
            )
            
            # Wrap in LangChain HuggingFacePipeline
            llm = HuggingFacePipeline(pipeline=pipe)
            print(f"LLM model {model_name} loaded successfully!")
            return llm
            
        except Exception as e:
            import traceback
            print(f"Error initializing LLM with {model_name}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print("Falling back to GPT-2 (lightweight model)...")
            return self._initialize_fallback_llm()
    
    def _initialize_fallback_llm(self):
        """Initialize a lightweight fallback LLM (GPT-2)."""
        try:
            from transformers import pipeline as hf_pipeline
            
            print("Loading fallback LLM (GPT-2)...")
            
            pipe = hf_pipeline(
                "text-generation",
                model="gpt2",
                max_new_tokens=50,  # Very short for GPT-2
                temperature=settings.temperature,
                device=-1,  # CPU
                return_full_text=False,
                do_sample=True,
                pad_token_id=50256  # GPT-2 pad token
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            print("Fallback LLM (GPT-2) loaded successfully!")
            return llm
        except Exception as e:
            import traceback
            print(f"Failed to initialize fallback LLM: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize fallback LLM: {e}")
    
    def _truncate_context(self, context: str, max_tokens: int = 500) -> str:
        """
        Truncate context to fit within token limit.
        
        Args:
            context: Context string to truncate
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated context string
        """
        try:
            # Try to get tokenizer from the pipeline
            if hasattr(self.llm, 'pipeline') and hasattr(self.llm.pipeline, 'tokenizer'):
                tokenizer = self.llm.pipeline.tokenizer
                # Tokenize the context
                tokens = tokenizer.encode(context, add_special_tokens=False)
                
                # Truncate if too long
                if len(tokens) > max_tokens:
                    tokens = tokens[:max_tokens]
                    context = tokenizer.decode(tokens, skip_special_tokens=True)
                    context += "... [truncated]"
                    
            else:
                # Fallback: simple character-based truncation
                # Rough estimate: 1 token ≈ 4 characters
                max_chars = max_tokens * 4
                if len(context) > max_chars:
                    context = context[:max_chars]
                    context += "... [truncated]"
                    
        except Exception as e:
            print(f"Warning: Could not truncate context properly: {e}")
            # Fallback: simple truncation
            max_chars = max_tokens * 4
            if len(context) > max_chars:
                context = context[:max_chars] + "... [truncated]"
        
        return context
    
    def _call_llm_with_timeout(self, prompt: str, timeout: int = 30) -> str:
        """
        Call LLM with timeout handling.
        Tries direct pipeline call first, then falls back to LangChain invoke.
        
        Args:
            prompt: Input prompt
            timeout: Timeout in seconds
            
        Returns:
            Generated answer text
        """
        result_container = {"result": None, "error": None, "done": False}
        
        def call_llm():
            try:
                # Try direct pipeline call first (faster)
                if hasattr(self.llm, 'pipeline') and self.llm.pipeline is not None:
                    print("Using direct pipeline call...")
                    pipeline_result = self.llm.pipeline(
                        prompt,
                        max_new_tokens=50,
                        temperature=settings.temperature,
                        do_sample=True,
                        pad_token_id=50256,
                        return_full_text=False
                    )
                    
                    # Extract text from pipeline result
                    if isinstance(pipeline_result, list) and len(pipeline_result) > 0:
                        if isinstance(pipeline_result[0], dict):
                            result_container["result"] = pipeline_result[0].get("generated_text", "")
                        else:
                            result_container["result"] = str(pipeline_result[0])
                    else:
                        result_container["result"] = str(pipeline_result)
                else:
                    # Fallback to LangChain invoke
                    print("Using LangChain invoke...")
                    result = self.llm.invoke(prompt)
                    
                    # Handle different return types
                    if isinstance(result, str):
                        result_container["result"] = result
                    elif isinstance(result, dict):
                        result_container["result"] = result.get("text", result.get("content", str(result)))
                    elif hasattr(result, 'content'):
                        result_container["result"] = result.content
                    elif isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict):
                            result_container["result"] = result[0].get("generated_text", result[0].get("text", str(result[0])))
                        else:
                            result_container["result"] = str(result[0])
                    else:
                        result_container["result"] = str(result)
                        
            except Exception as e:
                result_container["error"] = str(e)
            finally:
                result_container["done"] = True
        
        # Start LLM call in a thread
        thread = threading.Thread(target=call_llm)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        # Check if we got a result
        if result_container["done"]:
            if result_container["error"]:
                raise Exception(f"LLM call failed: {result_container['error']}")
            if result_container["result"]:
                return result_container["result"]
            else:
                return "I couldn't generate a response. Please try again."
        else:
            # Timeout occurred
            print(f"WARNING: LLM call timed out after {timeout} seconds")
            # Return a simple summary based on context
            return f"Based on the document, here's what I found: {context[:200]}..."
    
    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a RAG query.
        
        Args:
            request: Query request with question and parameters
            
        Returns:
            Query response with answer and sources
        """
        print(f"\n=== Processing query: {request.query} ===")
        
        # Initialize variables
        retrieved_docs = []
        answer = ""
        
        try:
            # Retrieve relevant documents
            k = request.top_k or settings.top_k_retrieval
            print(f"Retrieving top {k} documents...")
            retrieved_docs = self.vector_store.similarity_search_with_scores(
                query=request.query,
                k=k
            )
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            if not retrieved_docs:
                return QueryResponse(
                    answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
                    sources=[],
                    query=request.query,
                    timestamp=datetime.now()
                )
            
            # Build context from retrieved documents - prioritize by relevance score
            # Sort by score (lower is better for distance, but we want higher similarity)
            sorted_docs = sorted(retrieved_docs, key=lambda x: x[1], reverse=False)  # Lower score = more similar
            
            # Extract question keywords to find most relevant chunks
            question_lower = request.query.lower()
            question_keywords = set([w for w in question_lower.split() if len(w) > 3])  # Words longer than 3 chars
            
            # Score each document chunk by how well it matches the question
            scored_chunks = []
            for doc, score in sorted_docs[:5]:  # Consider top 5
                doc_lower = doc.page_content.lower()
                # Count keyword matches
                keyword_matches = sum(1 for keyword in question_keywords if keyword in doc_lower)
                # Combine similarity score with keyword matches
                relevance_score = keyword_matches * 0.5 - score  # Higher is better
                scored_chunks.append((doc, score, relevance_score, keyword_matches))
            
            # Sort by relevance (most relevant first)
            scored_chunks.sort(key=lambda x: x[2], reverse=True)
            
            # Select top 2-3 most relevant chunks
            selected_chunks = scored_chunks[:3]
            
            # Build context from selected chunks
            context_parts = []
            for doc, score, rel_score, kw_matches in selected_chunks:
                doc_text = doc.page_content
                # Try to find the most relevant part of the chunk if it's long
                if len(doc_text) > 400 and question_keywords:
                    # Find sentences that contain question keywords
                    sentences = doc_text.split('.')
                    relevant_sentences = []
                    for sentence in sentences:
                        sentence_lower = sentence.lower()
                        if any(keyword in sentence_lower for keyword in question_keywords):
                            relevant_sentences.append(sentence.strip())
                    
                    if relevant_sentences:
                        doc_text = '. '.join(relevant_sentences[:3])  # Top 3 relevant sentences
                    else:
                        doc_text = doc_text[:400]  # Fallback to first 400 chars
                else:
                    doc_text = doc_text[:400]  # Limit length
                
                context_parts.append(doc_text)
            
            context = "\n\n".join(context_parts)
            print(f"Selected {len(selected_chunks)} most relevant chunks (keyword matches: {[c[3] for c in selected_chunks]})")
            
            # For now, provide a simple summary without LLM (GPT-2 is too slow/unreliable)
            # This ensures the system works while we can optimize LLM later
            print("Generating response from retrieved context...")
            
            # Create a better formatted answer from the context
            if context:
                # Clean up the context - remove excessive whitespace and format better
                cleaned_context = " ".join(context.split())
                
                # Try to extract a meaningful snippet (first 500 chars, complete sentences)
                context_snippet = cleaned_context[:500]
                # Try to end at a sentence boundary
                last_period = context_snippet.rfind('.')
                if last_period > 250:  # Only if we have enough content
                    context_snippet = context_snippet[:last_period + 1]
                else:
                    context_snippet = cleaned_context[:500] + "..."
                
                # Build a more natural, question-aware response
                answer = f"Based on your question about '{request.query}', here's what I found in the document(s):\n\n{context_snippet}"
                
                # Add source information
                if retrieved_docs:
                    unique_sources = set([doc.metadata.get('source', 'Unknown') for doc, score in retrieved_docs])
                    sources_list = ", ".join(unique_sources)
                    answer += f"\n\n📄 Source(s): {sources_list}"
            else:
                answer = "I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing your question or upload more relevant documents."
            
            print(f"Answer generated: {answer[:100]}...")
            
            # Optional: Try LLM in background (commented out for now due to performance)
            # Uncomment this if you want to try LLM generation (may be slow)
            # try:
            #     prompt = f"Context: {context}\n\nQ: {request.query}\nA:"
            #     llm_answer = self._call_llm_with_timeout(prompt, timeout=10)
            #     if llm_answer and len(llm_answer) > 20:  # Only use if we got a decent response
            #         answer = llm_answer
            #         print("Used LLM-generated answer")
            # except Exception as e:
            #     print(f"LLM generation failed, using context summary: {e}")
                
        except Exception as e:
            import traceback
            print(f"ERROR in query processing: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            answer = f"I apologize, but I encountered an error while processing your question. Please try rephrasing it or check the backend logs for details."
        
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
        
        print(f"Returning response with {len(sources)} sources")
        
        return QueryResponse(
            answer=answer.strip() if answer else "I couldn't generate a response. Please try again.",
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