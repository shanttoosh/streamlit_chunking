# API Client for UI
import requests
import streamlit as st
from typing import Dict, Any, Optional
import json

class APIClient:
    """API client for communicating with the FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def _make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None, files: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP request to API"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = requests.get(url, timeout=30)
            elif method.upper() == "POST":
                if files:
                    response = requests.post(url, data=data, files=files, timeout=30)
                else:
                    response = requests.post(url, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return self._make_request("GET", "/health")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return self._make_request("GET", "/api/v1/system_info")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get system capabilities"""
        return self._make_request("GET", "/api/v1/capabilities")
    
    def process_fast(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with Fast Mode"""
        # Prepare form data
        form_data = {k: v for k, v in data.items() if k != "file"}
        files = {"file": data["file"]} if "file" in data else None
        
        return self._make_request("POST", "/api/v1/process/fast", data=form_data, files=files)
    
    def process_config1(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with Config-1 Mode"""
        # Prepare form data
        form_data = {k: v for k, v in data.items() if k != "file"}
        files = {"file": data["file"]} if "file" in data else None
        
        return self._make_request("POST", "/api/v1/process/config1", data=form_data, files=files)
    
    def process_deep_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with Deep Config Mode"""
        # Prepare form data
        form_data = {k: v for k, v in data.items() if k != "file"}
        files = {"file": data["file"]} if "file" in data else None
        
        return self._make_request("POST", "/api/v1/process/deep_config", data=form_data, files=files)
    
    def retrieve_similar(self, query: str, k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve similar chunks"""
        data = {"query": query, "k": k}
        if metadata_filter:
            data["metadata_filter"] = json.dumps(metadata_filter)
        
        return self._make_request("POST", "/api/v1/retrieve", data=data)
    
    def export_chunks(self) -> Dict[str, Any]:
        """Export chunks"""
        return self._make_request("GET", "/api/v1/export/chunks")
    
    def export_embeddings(self) -> Dict[str, Any]:
        """Export embeddings"""
        return self._make_request("GET", "/api/v1/export/embeddings")
    
    def export_embeddings_text(self) -> Dict[str, Any]:
        """Export embeddings as text"""
        return self._make_request("GET", "/api/v1/export/embeddings_text")
    
    def test_database_connection(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test database connection"""
        return self._make_request("POST", "/api/v1/db/test_connection", data=db_config)
    
    def list_database_tables(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """List database tables"""
        return self._make_request("POST", "/api/v1/db/list_tables", data=db_config)
    
    def import_database_table(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Import database table"""
        return self._make_request("POST", "/api/v1/db/import_one", data=db_config)
