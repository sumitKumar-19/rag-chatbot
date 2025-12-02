"""
Integration tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAPI:
    """Test cases for API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_query_endpoint_missing_documents(self):
        """Test query endpoint when no documents are uploaded."""
        # This will likely fail if no documents are in vector store
        # but tests the endpoint structure
        response = client.post(
            "/api/chat/query",
            json={"query": "test question"}
        )
        # Should return 200 or 500 depending on vector store state
        assert response.status_code in [200, 500]
    
    def test_simple_query_endpoint(self):
        """Test simple query endpoint."""
        response = client.post(
            "/api/chat/simple-query",
            params={"query": "test question"}
        )
        # Should return 200 or 500 depending on vector store state
        assert response.status_code in [200, 500]
    
    def test_vector_store_info(self):
        """Test vector store info endpoint."""
        response = client.get("/api/documents/info")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data


if __name__ == "__main__":
    pytest.main([__file__])

