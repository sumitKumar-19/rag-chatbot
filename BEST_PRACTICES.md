# RAG Chatbot - Best Practices Guide

This document outlines best practices for building, deploying, and maintaining a production-ready RAG chatbot.

## 📋 Table of Contents

1. [Prompt Engineering](#prompt-engineering)
2. [Document Processing](#document-processing)
3. [Embedding & Retrieval](#embedding--retrieval)
4. [Cost Control](#cost-control)
5. [Evaluation & Testing](#evaluation--testing)
6. [Performance Optimization](#performance-optimization)
7. [Security & Privacy](#security--privacy)

---

## 🎯 Prompt Engineering

### System Prompts

**Best Practice:** Define clear system prompts that set the assistant's role and behavior.

```python
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided context.
- Only use information from the provided context
- If the answer is not in the context, say so
- Be concise and accurate
- Cite sources when possible"""
```

### Context Formatting

**Best Practice:** Structure context clearly with metadata.

```python
context = "\n\n".join([
    f"[Source: {source}, Page: {page}]\n{content}"
    for source, page, content in retrieved_chunks
])
```

### Few-Shot Examples

**Best Practice:** Include examples in prompts to guide model behavior.

```python
prompt = f"""Examples:
Q: What is the capital of France?
A: The capital of France is Paris.

Q: {user_question}
A:"""
```

### Prompt Templates

**Best Practice:** Use parameterized templates for consistency.

```python
from langchain.prompts import PromptTemplate

template = """Context: {context}
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
```

---

## 📄 Document Processing

### Chunking Strategies

**1. Recursive Character Text Splitter (Recommended)**
- Preserves semantic meaning
- Handles various document structures
- Configurable chunk size and overlap

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**2. Semantic Chunking (Advanced)**
- Split based on semantic similarity
- Better for preserving context
- More computationally expensive

**Best Practices:**
- **Chunk Size:** 500-1500 tokens (balance context vs. precision)
- **Overlap:** 10-20% of chunk size (maintains context continuity)
- **Preserve Metadata:** Include source, page number, timestamp

### Document Preprocessing

**Best Practice:** Clean and normalize text before chunking.

```python
def preprocess_text(text: str) -> str:
    # Remove excessive whitespace
    text = " ".join(text.split())
    # Remove special characters if needed
    # Normalize unicode
    return text
```

---

## 🔍 Embedding & Retrieval

### Embedding Models

**Best Practice:** Choose models based on:
- **Speed:** Smaller models (all-MiniLM-L6-v2) for real-time
- **Quality:** Larger models (all-mpnet-base-v2) for accuracy
- **Language:** Multilingual models for non-English content

### Retrieval Strategies

**1. Similarity Search (Current Implementation)**
```python
results = vectorstore.similarity_search(query, k=5)
```

**2. MMR (Maximum Marginal Relevance) - Recommended**
- Balances relevance and diversity
- Reduces redundant results

```python
results = vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=20)
```

**3. Hybrid Search (Advanced)**
- Combine semantic and keyword search
- Better for specific queries

### Top-K Selection

**Best Practice:** 
- Start with k=5 for most queries
- Increase to k=10 for complex questions
- Monitor retrieval quality vs. latency

---

## 💰 Cost Control

### Model Selection

**Free Tier (Hugging Face):**
- ✅ No API costs
- ✅ Full control
- ⚠️ Requires local compute
- ⚠️ Slower inference

**Paid APIs (OpenAI, Anthropic):**
- ✅ Fast inference
- ✅ High quality
- ⚠️ Per-token costs
- ⚠️ Rate limits

### Optimization Strategies

**1. Caching**
```python
# Cache embeddings for unchanged documents
@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    return embedding_service.embed_text(text)
```

**2. Batch Processing**
```python
# Process multiple texts at once
embeddings = embedding_service.embed_documents(texts)  # More efficient
```

**3. Token Limits**
```python
# Limit context window
MAX_CONTEXT_TOKENS = 4000
truncated_context = context[:MAX_CONTEXT_TOKENS]
```

**4. Model Quantization**
```python
# Use 8-bit quantization to reduce memory
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

---

## 🧪 Evaluation & Testing

### Retrieval Metrics

**1. Precision@K**
- Percentage of retrieved documents that are relevant
- Target: >0.7

**2. Recall@K**
- Percentage of relevant documents retrieved
- Target: >0.8

**3. MRR (Mean Reciprocal Rank)**
- Average of reciprocal ranks of first relevant result
- Target: >0.6

### Answer Quality Metrics

**1. Faithfulness**
- Answer is grounded in context
- No hallucinations

**2. Relevance**
- Answer addresses the question
- Complete and accurate

**3. Latency**
- Response time < 3 seconds (user-facing)
- Target: < 1 second (optimal)

### Evaluation Framework

```python
def evaluate_rag(query: str, expected_answer: str, retrieved_docs: List) -> dict:
    """Evaluate RAG system performance."""
    metrics = {
        "retrieval_precision": calculate_precision(retrieved_docs),
        "answer_similarity": calculate_similarity(generated_answer, expected_answer),
        "latency": measure_latency(),
        "faithfulness": check_faithfulness(generated_answer, retrieved_docs)
    }
    return metrics
```

---

## ⚡ Performance Optimization

### Vector Store Optimization

**1. Indexing**
- Use HNSW (Hierarchical Navigable Small World) for fast search
- ChromaDB uses HNSW by default

**2. Batch Operations**
```python
# Add documents in batches
vectorstore.add_documents(documents, batch_size=100)
```

**3. Persistence**
- Persist vector store to disk
- Load on startup, not per-query

### LLM Optimization

**1. Model Caching**
```python
# Cache loaded models
from transformers import pipeline
pipe = pipeline("text-generation", model="model_name")
# Reuse pipe instance
```

**2. Streaming Responses**
```python
# Stream tokens as they're generated
for token in llm.stream(prompt):
    yield token
```

**3. Async Processing**
```python
# Use async for concurrent requests
async def process_query(query: str):
    result = await asyncio.to_thread(rag_service.query, query)
    return result
```

---

## 🔒 Security & Privacy

### Data Privacy

**Best Practice:**
- ✅ Store documents locally
- ✅ Use local models when possible
- ✅ Encrypt sensitive documents
- ✅ Implement access controls

### Input Validation

```python
def validate_query(query: str) -> bool:
    """Validate user query."""
    if len(query) > 1000:
        raise ValueError("Query too long")
    if not query.strip():
        raise ValueError("Empty query")
    # Check for injection attempts
    if any(char in query for char in ["<", ">", "script"]):
        raise ValueError("Invalid characters")
    return True
```

### Rate Limiting

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@router.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, ...):
    ...
```

---

## 📊 Monitoring & Logging

### Logging Best Practices

```python
import logging

logger = logging.getLogger(__name__)

def query_rag(query: str):
    logger.info(f"Processing query: {query[:50]}...")
    start_time = time.time()
    
    try:
        result = rag_service.query(query)
        latency = time.time() - start_time
        logger.info(f"Query completed in {latency:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise
```

### Metrics to Track

- Query latency (p50, p95, p99)
- Token usage per query
- Retrieval quality scores
- Error rates
- User satisfaction (if available)

---

## 🚀 Deployment Considerations

### Environment Configuration

**Best Practice:** Use environment variables for all configuration.

```python
# .env file
MODEL_PATH=/models/llm
VECTOR_DB_PATH=/data/vectorstore
MAX_CONCURRENT_REQUESTS=10
```

### Containerization

**Dockerfile Best Practices:**
- Use multi-stage builds
- Cache model downloads
- Set resource limits
- Use non-root user

### Scaling

**Horizontal Scaling:**
- Stateless API (FastAPI)
- Shared vector store (Qdrant, Pinecone)
- Load balancer

**Vertical Scaling:**
- More RAM for larger models
- GPU for faster inference
- SSD for vector store

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Vector Database Guide](https://www.pinecone.io/learn/vector-database/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

---

## 🎓 Learning Path

1. **Start Simple:** Basic RAG with small documents
2. **Iterate:** Add features incrementally
3. **Measure:** Track metrics from day one
4. **Optimize:** Focus on bottlenecks
5. **Scale:** Plan for growth

Remember: **Perfect is the enemy of good.** Start with an MVP and iterate based on real usage patterns!

