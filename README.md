# RAG Chatbot - Production-Style Learning Project

A comprehensive Retrieval-Augmented Generation (RAG) chatbot built with LangChain, designed as a learning project for understanding LLMs and RAG systems end-to-end.

## 🏗️ Architecture

### Tech Stack

**Backend:**
- **FastAPI** - Modern, fast Python web framework (similar to Spring Boot in philosophy)
- **LangChain** - LLM orchestration and RAG pipeline
- **LangChain Community** - Document loaders and integrations

**Embeddings & LLM:**
- **Hugging Face** (default, free tier) - Sentence transformers for embeddings, open-source LLMs for generation
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (lightweight, fast)
  - LLM: `microsoft/DialoGPT-medium` or `google/flan-t5-base` (CPU-friendly)
  - Can run locally without API costs

**Vector Database:**
- **Chroma** (MVP) - Lightweight, in-memory vector store
- **Qdrant** (Production) - Scalable, persistent vector database

**Frontend:**
- **Streamlit** (MVP) - Rapid prototyping
- **React/Next.js** (Future) - Production frontend

**Additional:**
- **PyPDF2** / **pypdf** - PDF document processing
- **python-dotenv** - Environment configuration
- **pytest** - Testing framework

## 📋 Project Roadmap

### Phase 1: MVP (Week 1-2)
- [x] Project setup and architecture
- [ ] PDF document ingestion pipeline
- [ ] Text chunking and embedding generation
- [ ] Vector store integration
- [ ] Basic RAG query endpoint
- [ ] Simple Streamlit UI

### Phase 2: Core Features (Week 3-4)
- [ ] Multi-document support
- [ ] Chat history and conversation context
- [ ] Source citation and retrieval metadata
- [ ] Document management (upload/delete)
- [ ] Error handling and validation

### Phase 3: Enhancement (Week 5-6)
- [ ] Advanced chunking strategies (semantic, recursive)
- [ ] Query optimization (query rewriting, reranking)
- [ ] Prompt templates and system prompts
- [ ] Response streaming
- [ ] Performance monitoring

### Phase 4: Testing & Quality (Week 7)
- [ ] Unit tests for core components
- [ ] Integration tests for API endpoints
- [ ] RAG evaluation metrics (retrieval accuracy, answer quality)
- [ ] Load testing

### Phase 5: Deployment (Week 8)
- [ ] Docker containerization
- [ ] Environment configuration
- [ ] Cloud deployment guide (AWS/GCP/Azure)
- [ ] CI/CD pipeline setup

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Hugging Face account (free) - Get token from https://huggingface.co/settings/tokens
- 4GB+ RAM recommended (for local models)
- Optional: GPU for faster inference (CUDA-compatible)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Hugging Face token (optional, but recommended for private models)
```

### Run Application

```bash
# Start FastAPI backend
uvicorn app.main:app --reload

# Start Streamlit frontend (in another terminal)
streamlit run app/frontend/streamlit_app.py
```

## 📁 Project Structure

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   ├── document.py
│   │   └── query.py
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── document_service.py # Document ingestion
│   │   ├── embedding_service.py # Embedding generation
│   │   ├── vector_store.py     # Vector DB operations
│   │   └── rag_service.py      # RAG pipeline
│   ├── api/                    # API routes
│   │   ├── __init__.py
│   │   ├── documents.py
│   │   └── chat.py
│   └── frontend/               # Frontend interfaces
│       └── streamlit_app.py
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_document_service.py
│   ├── test_rag_service.py
│   └── test_api.py
├── data/                       # Document storage
│   └── uploads/
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
└── README.md                   # This file
```

## 🎯 Key Concepts & Best Practices

### 1. Document Processing Pipeline
- **Loading**: PDF → Text extraction
- **Chunking**: Text → Semantic chunks (overlap for context)
- **Embedding**: Chunks → Vector representations
- **Storage**: Vectors → Vector database with metadata

### 2. RAG Query Flow
1. User query → Embedding
2. Similarity search in vector store
3. Retrieve top-k relevant chunks
4. Construct prompt with context
5. LLM generation with retrieved context
6. Return response with citations

### 3. Prompt Engineering
- **System Prompt**: Define assistant role and behavior
- **Context Injection**: Include retrieved documents
- **Query Formatting**: Structure for clarity
- **Few-shot Examples**: Guide model behavior

### 4. Cost Control
- Use appropriate model tiers (GPT-3.5-turbo vs GPT-4)
- Cache embeddings (don't re-embed unchanged documents)
- Limit context window size
- Monitor token usage

### 5. Evaluation Metrics
- **Retrieval Accuracy**: Are relevant chunks retrieved?
- **Answer Quality**: Is the answer correct and complete?
- **Latency**: Response time
- **Cost per Query**: Token usage tracking

## 📚 Learning Resources

- [LangChain Documentation](https://python.langchain.com/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Vector Database Guide](https://www.pinecone.io/learn/vector-database/)

## 🤝 Contributing

This is a learning project. Feel free to experiment, modify, and extend!

## 📄 License

MIT License
