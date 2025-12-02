"""
Streamlit frontend for the RAG Chatbot.
Simple, interactive UI for document upload and Q&A.
"""
import streamlit as st
import requests
from typing import Optional
import time

# Configuration
API_BASE_URL = "http://localhost:8001"

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


def upload_document(file) -> Optional[dict]:
    """Upload a document to the backend."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(
            f"{API_BASE_URL}/api/documents/upload",
            files=files
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return None


def query_rag(question: str, top_k: Optional[int] = None) -> Optional[dict]:
    """Send a query to the RAG API."""
    try:
        payload = {
            "query": question,
            "top_k": top_k
        }
        response = requests.post(
            f"{API_BASE_URL}/api/chat/simple-query",
            params=payload
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying: {str(e)}")
        return None


def get_vector_store_info() -> Optional[dict]:
    """Get vector store information."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/documents/info")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error getting info: {str(e)}")
        return None


# Main UI
st.title("🤖 RAG Chatbot")
st.markdown("Upload PDF documents and ask questions based on their content!")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Management")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a PDF file to add it to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("Upload Document"):
            with st.spinner("Processing document..."):
                result = upload_document(uploaded_file)
                if result:
                    st.success(f"✅ {result['message']}")
                    st.json(result)
    
    st.divider()
    
    # Vector store info
    if st.button("📊 View Knowledge Base Info"):
        info = get_vector_store_info()
        if info:
            st.json(info)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.header("💬 Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:** {source['source']}")
                    if source.get("page_number"):
                        st.markdown(f"*Page: {source['page_number']}*")
                    st.markdown(f"*Similarity: {source['score']:.3f}*")
                    st.markdown(f"```\n{source['content']}\n```")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_rag(prompt)
            
            if response:
                answer = response.get("answer", "No answer generated")
                sources = response.get("sources", [])
                
                st.markdown(answer)
                
                # Display sources
                if sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {source['source']}")
                            if source.get("page_number"):
                                st.markdown(f"*Page: {source['page_number']}*")
                            st.markdown(f"*Similarity: {source['score']:.3f}*")
                            st.markdown(f"```\n{source['content']}\n```")
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            else:
                st.error("Failed to get response from the API")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>RAG Chatbot - Built with LangChain & Hugging Face</p>
    </div>
    """,
    unsafe_allow_html=True
)

