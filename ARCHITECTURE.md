# RAG Chatbot - Architecture Documentation

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────┐
│   Frontend  │  Streamlit UI / React
│  (Streamlit)│
└──────┬──────┘
       │ HTTP/REST
       │
┌──────▼─────────────────────────────────────┐
│         FastAPI Backend                     │
│  ┌──────────────────────────────────────┐  │
│  │  API Layer (documents, chat)         │  │
│  └───────────┬──────────────────────────┘  │
│              │                              │
│  ┌───────────▼──────────────────────────┐  │
│  │  Service Layer                       │  │
│  │  - DocumentService                   │  │
│  │  - EmbeddingService                  │  │
│  │  - VectorStoreService                │  │
│  │  - RAGService                        │  │
│  └───────────┬──────────────────────────┘  │
└──────────────┼─────────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼───┐  ┌───▼────┐
│Chroma │  │Hugging│  │File    │
│Vector │  │Face   │  │Storage │
│Store  │  │Models │  │        │
└───────┘  └───────┘  └────────┘
```

## 📦 Component Details

### 1. Frontend Layer

**Streamlit Application** (`app/frontend/streamlit_app.py`)
- **Purpose:** User interface for document upload and chat
- **Features:**
  - PDF file upload
  - Interactive chat interface
  - Source citation display
  - Vector store statistics

**Future:** React/Next.js frontend for production

### 2. API Layer

**FastAPI Application** (`app/main.py`)
- **Framework:** FastAPI (async, fast, auto-docs)
- **Endpoints:**
  - `GET /` - Root endpoint
  - `GET /health` - Health check
  - `POST /api/documents/upload` - Upload PDF
  - `GET /api/documents/info` - Vector store info
  - `DELETE /api/documents/{id}` - Delete document
  - `POST /api/chat/query` - RAG query
  - `POST /api/chat/simple-query` - Simplified query

### 3. Service Layer

#### DocumentService (`app/services/document_service.py`)
- **Responsibilities:**
  - PDF loading and text extraction
  - Document chunking
  - File management
- **Dependencies:** PyPDFLoader, RecursiveCharacterTextSplitter

#### EmbeddingService (`app/services/embedding_service.py`)
- **Responsibilities:**
  - Text embedding generation
  - Batch processing
  - Model management
- **Dependencies:** HuggingFaceEmbeddings, SentenceTransformers

#### VectorStoreService (`app/services/vector_store.py`)
- **Responsibilities:**
  - Vector storage and retrieval
  - Similarity search
  - Document management
- **Dependencies:** ChromaDB, LangChain VectorStore

#### RAGService (`app/services/rag_service.py`)
- **Responsibilities:**
  - RAG pipeline orchestration
  - LLM integration
  - Query processing
- **Dependencies:** HuggingFacePipeline, RetrievalQA

### 4. Data Layer

#### Vector Database (ChromaDB)
- **Type:** In-memory with persistence
- **Storage:** Local filesystem (`./data/vectorstore`)
- **Features:**
  - HNSW indexing for fast search
  - Metadata filtering
  - Persistence to disk

#### File Storage
- **Location:** `./data/uploads`
- **Format:** PDF files
- **Management:** Temporary storage (can be cleaned)

## 🔄 Data Flow

### Document Ingestion Flow

```
1. User uploads PDF
   ↓
2. DocumentService.load_pdf()
   - Extract text from PDF
   ↓
3. DocumentService.chunk_documents()
   - Split into chunks with overlap
   - Add metadata (source, page, etc.)
   ↓
4. EmbeddingService.embed_documents()
   - Generate embeddings for each chunk
   ↓
5. VectorStoreService.add_documents()
   - Store vectors in ChromaDB
   - Index for fast retrieval
```

### Query Flow

```
1. User submits query
   ↓
2. EmbeddingService.embed_text(query)
   - Generate query embedding
   ↓
3. VectorStoreService.similarity_search()
   - Find top-k similar chunks
   - Return documents with scores
   ↓
4. RAGService.query()
   - Build context from retrieved chunks
   - Format prompt with context
   ↓
5. LLM generation
   - Generate answer based on context
   ↓
6. Format response
   - Include answer + source citations
   - Return to user
```

## 🔧 Technology Stack

### Backend
- **FastAPI:** Modern Python web framework
- **LangChain:** LLM orchestration
- **ChromaDB:** Vector database
- **Hugging Face:** Models and embeddings

### Models
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
  - Dimension: 384
  - Fast, lightweight
  - Good quality for most use cases

- **LLM:** `microsoft/DialoGPT-medium` (default)
  - Small, CPU-friendly
  - Alternative: `google/flan-t5-base`

### Frontend
- **Streamlit:** Rapid prototyping
- **Future:** React/Next.js for production

## 📁 Project Structure

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Configuration
│   ├── models/                 # Data models
│   │   ├── document.py
│   │   └── query.py
│   ├── services/               # Business logic
│   │   ├── document_service.py
│   │   ├── embedding_service.py
│   │   ├── vector_store.py
│   │   └── rag_service.py
│   ├── api/                    # API routes
│   │   ├── documents.py
│   │   └── chat.py
│   └── frontend/               # Frontend
│       └── streamlit_app.py
├── tests/                      # Tests
├── data/                       # Data storage
│   ├── uploads/               # Uploaded PDFs
│   └── vectorstore/           # ChromaDB data
├── requirements.txt
├── .env.example
└── README.md
```

## 🔐 Configuration Management

**Environment Variables** (`.env`)
- Model configuration
- API keys (optional for Hugging Face)
- Paths and directories
- RAG parameters

**Settings Class** (`app/config.py`)
- Type-safe configuration
- Default values
- Environment variable loading

## 🚀 Deployment Architecture

### Development
```
Local Machine
├── FastAPI (uvicorn --reload)
├── Streamlit (streamlit run)
└── ChromaDB (local filesystem)
```

### Production (Recommended)
```
Cloud Server / Container
├── FastAPI (gunicorn + uvicorn workers)
├── React Frontend (Nginx)
├── ChromaDB (persistent volume)
└── Model Cache (persistent volume)
```

### Scaling Options
1. **Horizontal:** Multiple FastAPI instances + load balancer
2. **Vertical:** More RAM/GPU for larger models
3. **Vector DB:** Migrate to Qdrant/Pinecone for distributed storage

## 🔄 Error Handling

**Service Layer:**
- Try-catch blocks
- Meaningful error messages
- Fallback mechanisms (e.g., GPT-2 fallback)

**API Layer:**
- HTTP status codes
- Error response models
- Input validation

**Frontend:**
- User-friendly error messages
- Loading states
- Retry mechanisms

## 📊 Performance Considerations

### Latency Targets
- **Document Upload:** < 5 seconds (depends on size)
- **Query Response:** < 3 seconds (user-facing)
- **Embedding Generation:** < 1 second per chunk

### Optimization Strategies
1. **Caching:** Embed embeddings, cache models
2. **Batch Processing:** Process multiple chunks at once
3. **Async Operations:** Use async/await for I/O
4. **Model Quantization:** Reduce model size

## 🔒 Security Considerations

1. **Input Validation:** Sanitize user inputs
2. **File Upload:** Validate file types and sizes
3. **Rate Limiting:** Prevent abuse
4. **Data Privacy:** Local storage, no external APIs (optional)
5. **Access Control:** Implement authentication (future)

## 📈 Future Enhancements

1. **Multi-document Support:** Already implemented
2. **Chat History:** Conversation context
3. **Advanced Retrieval:** MMR, reranking
4. **Streaming Responses:** Real-time token streaming
5. **Evaluation Framework:** Automated testing
6. **Monitoring:** Metrics and logging
7. **Authentication:** User management
8. **Multi-format Support:** DOCX, TXT, etc.

