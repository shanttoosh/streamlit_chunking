# Integration Tests for API
import pytest
import requests
import pandas as pd
import tempfile
import os
from fastapi.testclient import TestClient
from src.api.main import app

class TestAPI:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_csv_file(self, sample_dataframe):
        """Create sample CSV file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_dataframe.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_system_info(self, client):
        """Test system info endpoint"""
        response = client.get("/api/v1/system_info")
        assert response.status_code == 200
        data = response.json()
        assert "memory_usage" in data
        assert "available_memory" in data
    
    def test_capabilities(self, client):
        """Test capabilities endpoint"""
        response = client.get("/api/v1/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "large_file_support" in data
        assert "performance_features" in data
    
    def test_file_info_empty(self, client):
        """Test file info endpoint with no file"""
        response = client.get("/api/v1/file_info")
        assert response.status_code == 200
        data = response.json()
        assert data == {}
    
    def test_process_fast_with_file(self, client, sample_csv_file):
        """Test fast processing with file upload"""
        with open(sample_csv_file, 'rb') as f:
            files = {"file": ("test.csv", f, "text/csv")}
            data = {
                "use_openai": False,
                "use_turbo": True,
                "batch_size": 256
            }
            response = client.post("/api/v1/process/fast", files=files, data=data)
        
        assert response.status_code == 200
        data = response.json()
        assert "rows" in data
        assert "chunks" in data
        assert "stored" in data
        assert "embedding_model" in data
        assert "retrieval_ready" in data
    
    def test_process_config1_with_file(self, client, sample_csv_file):
        """Test config1 processing with file upload"""
        with open(sample_csv_file, 'rb') as f:
            files = {"file": ("test.csv", f, "text/csv")}
            data = {
                "chunk_method": "recursive",
                "chunk_size": 400,
                "overlap": 50,
                "model_choice": "paraphrase-MiniLM-L6-v2",
                "storage_choice": "faiss",
                "retrieval_metric": "cosine",
                "use_openai": False,
                "use_turbo": False,
                "batch_size": 256
            }
            response = client.post("/api/v1/process/config1", files=files, data=data)
        
        assert response.status_code == 200
        data = response.json()
        assert "rows" in data
        assert "chunks" in data
        assert "stored" in data
        assert "embedding_model" in data
        assert "retrieval_ready" in data
    
    def test_process_deep_config_with_file(self, client, sample_csv_file):
        """Test deep config processing with file upload"""
        with open(sample_csv_file, 'rb') as f:
            files = {"file": ("test.csv", f, "text/csv")}
            data = {
                "preprocessing_config": "{}",
                "chunking_config": '{"method": "fixed", "chunk_size": 400, "overlap": 50}',
                "embedding_config": '{"model_name": "paraphrase-MiniLM-L6-v2", "batch_size": 64}',
                "storage_config": '{"type": "chroma", "collection_name": "test_collection"}'
            }
            response = client.post("/api/v1/process/deep_config", files=files, data=data)
        
        assert response.status_code == 200
        data = response.json()
        assert "rows" in data
        assert "chunks" in data
        assert "stored" in data
        assert "embedding_model" in data
        assert "retrieval_ready" in data
    
    def test_process_fast_no_file(self, client):
        """Test fast processing without file"""
        data = {
            "use_openai": False,
            "use_turbo": True,
            "batch_size": 256
        }
        response = client.post("/api/v1/process/fast", data=data)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_retrieve_no_model(self, client):
        """Test retrieval without model"""
        data = {
            "query": "test query",
            "k": 5
        }
        response = client.post("/api/v1/retrieve", data=data)
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
    
    def test_export_chunks_no_data(self, client):
        """Test export chunks without data"""
        response = client.get("/api/v1/export/chunks")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
    
    def test_export_embeddings_no_data(self, client):
        """Test export embeddings without data"""
        response = client.get("/api/v1/export/embeddings")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
    
    def test_database_test_connection_invalid(self, client):
        """Test database connection with invalid config"""
        data = {
            "db_type": "mysql",
            "host": "invalid_host",
            "port": 3306,
            "username": "invalid_user",
            "password": "invalid_pass",
            "database": "invalid_db"
        }
        response = client.post("/api/v1/db/test_connection", data=data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] == False
        assert "message" in data
    
    def test_database_list_tables_invalid(self, client):
        """Test database list tables with invalid config"""
        data = {
            "db_type": "mysql",
            "host": "invalid_host",
            "port": 3306,
            "username": "invalid_user",
            "password": "invalid_pass",
            "database": "invalid_db"
        }
        response = client.post("/api/v1/db/list_tables", data=data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        assert "message" in data
    
    def test_database_import_invalid(self, client):
        """Test database import with invalid config"""
        data = {
            "db_type": "mysql",
            "host": "invalid_host",
            "port": 3306,
            "username": "invalid_user",
            "password": "invalid_pass",
            "database": "invalid_db",
            "table_name": "invalid_table"
        }
        response = client.post("/api/v1/db/import_one", data=data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        assert "message" in data
