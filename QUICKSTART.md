# Quick Start Guide

Get your RAG chatbot up and running in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- 4GB+ RAM (for running models locally)
- Internet connection (for downloading models on first run)

## Step 1: Setup Environment

```bash
# Clone the repository (if not already done)
cd rag-chatbot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file (optional - defaults work for most cases)
# The only required setting is HUGGINGFACE_API_TOKEN if you want to use private models
```

**Note:** Hugging Face token is optional. Public models work without it, but you may hit rate limits. Get a free token at https://huggingface.co/settings/tokens

## Step 3: Start the Backend

```bash
# Start FastAPI server
uvicorn app.main:app --reload

# You should see:
# INFO:     Uvicorn running on http://127.0.0.1:8000
```

The API will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Step 4: Start the Frontend (New Terminal)

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start Streamlit
streamlit run app/frontend/streamlit_app.py

# You should see:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

## Step 5: Use the Application

1. **Open Browser:** Navigate to http://localhost:8501

2. **Upload a Document:**
   - Click "Browse files" in the sidebar
   - Select a PDF file
   - Click "Upload Document"
   - Wait for processing (first time may take longer as models download)

3. **Ask Questions:**
   - Type your question in the chat input
   - Press Enter or click Send
   - View the answer with source citations

## Testing the API Directly

You can also test the API using curl or the interactive docs:

### Upload a Document
```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

### Query the RAG System
```bash
curl -X POST "http://localhost:8000/api/chat/simple-query?query=What%20is%20this%20document%20about?" \
  -H "accept: application/json"
```

### View API Documentation
Open http://localhost:8000/docs in your browser for interactive API documentation.

## Troubleshooting

### Issue: Models take too long to download
**Solution:** Models download automatically on first use. This is normal and only happens once. The first query may take 1-2 minutes.

### Issue: Out of memory errors
**Solution:** 
- Use smaller models (change `LLM_MODEL` in `.env` to `gpt2`)
- Reduce `CHUNK_SIZE` in `.env`
- Close other applications

### Issue: Import errors
**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Port already in use
**Solution:**
```bash
# Change port in .env or use different port
uvicorn app.main:app --port 8001
```

## Next Steps

1. **Read the Documentation:**
   - `README.md` - Project overview
   - `ARCHITECTURE.md` - System architecture
   - `BEST_PRACTICES.md` - Best practices guide

2. **Experiment:**
   - Try different chunk sizes
   - Test with various PDF documents
   - Adjust temperature and top_k parameters

3. **Extend:**
   - Add more document formats (DOCX, TXT)
   - Implement chat history
   - Add authentication

## Example Workflow

```python
# Example: Using the API programmatically
import requests

# 1. Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/documents/upload",
        files={"file": f}
    )
print(response.json())

# 2. Ask a question
response = requests.post(
    "http://localhost:8000/api/chat/simple-query",
    params={"query": "What is the main topic of this document?"}
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} documents retrieved")
```

## Support

- Check the documentation files
- Review error messages in the terminal
- Test with a simple PDF first
- Ensure all dependencies are installed

Happy coding! 🚀

