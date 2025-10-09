# API Client for UI
import requests
import json
import tempfile
import os
from typing import Dict, Any, Optional

# FastAPI backend URL
API_BASE_URL = "http://localhost:8001"

class APIClient:
    """API client for making requests to the backend"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
    
    def call_fast_api(self, file_path: str, filename: str, db_type: str, db_config: dict = None, 
                     use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                     process_large_files: bool = True, use_turbo: bool = False, batch_size: int = 256):
        """Send CSV upload or trigger DB import for Fast mode"""
        try:
            # DB import path: send only form data (no file open)
            if db_config and db_config.get('use_db'):
                data = {
                    "db_type": db_config.get("db_type"),
                    "host": db_config.get("host"),
                    "port": db_config.get("port"),
                    "username": db_config.get("username"),
                    "password": db_config.get("password"),
                    "database": db_config.get("database"),
                    "table_name": db_config.get("table_name"),
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files,
                    "use_turbo": use_turbo,
                    "batch_size": batch_size
                }
                response = requests.post(f"{self.base_url}/api/v1/run_fast", data=data)
                return response.json()

            # CSV upload path: open and send file
            with open(file_path, 'rb') as f:
                files = {"file": (filename, f, "text/csv")}
                data = {
                    "db_type": db_type,
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files,
                    "use_turbo": use_turbo,
                    "batch_size": batch_size
                }
                response = requests.post(f"{self.base_url}/api/v1/run_fast", files=files, data=data)
            return response.json()
        except Exception as e:
            return {"error": f"API call failed: {str(e)}"}

    def call_config1_api(self, file_path: str, filename: str, config: dict, db_config: dict = None,
                        use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                        process_large_files: bool = True, use_turbo: bool = False, batch_size: int = 256):
        """Send CSV upload or trigger DB import for Config-1 mode"""
        try:
            # DB import path: send only form data (no file open)
            if db_config and db_config.get('use_db'):
                data = {
                    "db_type": db_config.get("db_type"),
                    "host": db_config.get("host"),
                    "port": db_config.get("port"),
                    "username": db_config.get("username"),
                    "password": db_config.get("password"),
                    "database": db_config.get("database"),
                    "table_name": db_config.get("table_name"),
                    "chunk_method": config.get("chunk_method", "recursive"),
                    "chunk_size": config.get("chunk_size", 400),
                    "overlap": config.get("overlap", 50),
                    "document_key_column": config.get("document_key_column"),
                    "token_limit": config.get("token_limit", 2000),
                    "retrieval_metric": config.get("retrieval_metric", "cosine"),
                    "model_choice": config.get("model_choice", "paraphrase-MiniLM-L6-v2"),
                    "storage_choice": config.get("storage_choice", "faiss"),
                    "apply_default_preprocessing": config.get("apply_default_preprocessing", True),
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files,
                    "use_turbo": use_turbo,
                    "batch_size": batch_size
                }
                response = requests.post(f"{self.base_url}/api/v1/run_config1", data=data)
                return response.json()

            # CSV upload path: open and send file
            with open(file_path, 'rb') as f:
                files = {"file": (filename, f, "text/csv")}
                data = {
                    "db_type": "sqlite",
                    "chunk_method": config.get("chunk_method", "recursive"),
                    "chunk_size": config.get("chunk_size", 400),
                    "overlap": config.get("overlap", 50),
                    "document_key_column": config.get("document_key_column"),
                    "token_limit": config.get("token_limit", 2000),
                    "retrieval_metric": config.get("retrieval_metric", "cosine"),
                    "model_choice": config.get("model_choice", "paraphrase-MiniLM-L6-v2"),
                    "storage_choice": config.get("storage_choice", "faiss"),
                    "apply_default_preprocessing": config.get("apply_default_preprocessing", True),
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files,
                    "use_turbo": use_turbo,
                    "batch_size": batch_size
                }
                response = requests.post(f"{self.base_url}/api/v1/run_config1", files=files, data=data)
            return response.json()
        except Exception as e:
            return {"error": f"Config1 API call failed: {str(e)}"}

    def call_deep_config_preprocess_api(self, file_path: str, filename: str, db_config: dict = None):
        """Step 1: Preprocess data"""
        try:
            if db_config and db_config.get('use_db'):
                data = {
                    "db_type": db_config.get("db_type"),
                    "host": db_config.get("host"),
                    "port": db_config.get("port"),
                    "username": db_config.get("username"),
                    "password": db_config.get("password"),
                    "database": db_config.get("database"),
                    "table_name": db_config.get("table_name")
                }
                response = requests.post(f"{self.base_url}/api/v1/deep_config/preprocess", data=data)
            else:
                with open(file_path, 'rb') as f:
                    files = {"file": (filename, f, "text/csv")}
                    response = requests.post(f"{self.base_url}/api/v1/deep_config/preprocess", files=files)
            return response.json()
        except Exception as e:
            return {"error": f"Preprocess API call failed: {str(e)}"}

    def call_deep_config_type_convert_api(self, type_conversions: dict):
        """Step 2: Convert data types"""
        try:
            data = {"type_conversions": json.dumps(type_conversions)}
            response = requests.post(f"{self.base_url}/api/v1/deep_config/type_convert", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"Type convert API call failed: {str(e)}"}

    def call_deep_config_null_handle_api(self, null_strategies: dict):
        """Step 3: Handle null values"""
        try:
            data = {"null_strategies": json.dumps(null_strategies)}
            response = requests.post(f"{self.base_url}/api/v1/deep_config/null_handle", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"Null handle API call failed: {str(e)}"}

    def call_deep_config_duplicates_api(self, strategy: str):
        """Step 3: Handle duplicate rows"""
        try:
            data = {"strategy": strategy}
            response = requests.post(f"{self.base_url}/api/v1/deep_config/duplicates", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"Duplicates API call failed: {str(e)}"}

    def call_deep_config_stopwords_api(self, remove_stopwords: bool):
        """Step 4: Remove stop words"""
        try:
            data = {"remove_stopwords": remove_stopwords}
            response = requests.post(f"{self.base_url}/api/v1/deep_config/stopwords", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"Stopwords API call failed: {str(e)}"}

    def call_deep_config_normalize_api(self, text_processing: str):
        """Step 5: Text normalization"""
        try:
            data = {"text_processing": text_processing}
            response = requests.post(f"{self.base_url}/api/v1/deep_config/normalize", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"Normalize API call failed: {str(e)}"}

    def call_deep_config_chunk_api(self, chunk_params: dict):
        """Step 6: Chunk data"""
        try:
            data = {"chunk_params": json.dumps(chunk_params)}
            response = requests.post(f"{self.base_url}/api/v1/deep_config/chunk", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"Chunk API call failed: {str(e)}"}

    def call_deep_config_embed_api(self, embed_params: dict):
        """Step 7: Generate embeddings"""
        try:
            data = {"embed_params": json.dumps(embed_params)}
            response = requests.post(f"{self.base_url}/api/v1/deep_config/embed", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"Embed API call failed: {str(e)}"}

    def call_deep_config_store_api(self, store_params: dict):
        """Step 8: Store embeddings"""
        try:
            data = {"store_params": json.dumps(store_params)}
            response = requests.post(f"{self.base_url}/api/v1/deep_config/store", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"Store API call failed: {str(e)}"}

    def call_retrieve_api(self, query: str, k: int = 5):
        """Semantic search"""
        try:
            data = {"query": query, "k": k}
            response = requests.post(f"{self.base_url}/api/v1/retrieve", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"Retrieve API call failed: {str(e)}"}

    def call_openai_retrieve_api(self, query: str, model: str = "all-MiniLM-L6-v2", n_results: int = 5):
        """OpenAI-compatible retrieval"""
        try:
            data = {"query": query, "model": model, "n_results": n_results}
            response = requests.post(f"{self.base_url}/api/v1/v1/retrieve", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"OpenAI retrieve API call failed: {str(e)}"}

    def call_openai_embeddings_api(self, text: str, model: str = "text-embedding-ada-002", 
                                  openai_api_key: str = None, openai_base_url: str = None):
        """OpenAI-compatible embeddings"""
        try:
            data = {
                "model": model,
                "input": text,
                "openai_api_key": openai_api_key,
                "openai_base_url": openai_base_url
            }
            response = requests.post(f"{self.base_url}/v1/embeddings", data=data)
            return response.json()
        except Exception as e:
            return {"error": f"OpenAI embeddings API call failed: {str(e)}"}

    def get_system_info_api(self):
        """Get system information"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/system_info")
            return response.json()
        except Exception as e:
            return {"error": f"System info API call failed: {str(e)}"}

    def get_file_info_api(self):
        """Get file information"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/file_info")
            return response.json()
        except Exception as e:
            return {"error": f"File info API call failed: {str(e)}"}

    def get_capabilities_api(self):
        """Get system capabilities"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/capabilities")
            return response.json()
        except Exception as e:
            return {"error": f"Capabilities API call failed: {str(e)}"}

    def db_test_connection_api(self, payload: dict):
        """Test database connection"""
        try:
            response = requests.post(f"{self.base_url}/api/v1/db/test_connection", data=payload)
            return response.json()
        except Exception as e:
            return {"error": f"DB test connection API call failed: {str(e)}"}

    def db_list_tables_api(self, payload: dict):
        """List database tables"""
        try:
            response = requests.post(f"{self.base_url}/api/v1/db/list_tables", data=payload)
            return response.json()
        except Exception as e:
            return {"error": f"DB list tables API call failed: {str(e)}"}

    def download_file(self, url: str, filename: str):
        """Download file from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            return None

    def download_embeddings_text(self):
        """Download embeddings as text"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/export/embeddings_text")
            return response.text
        except Exception as e:
            return None

# Legacy function wrappers for backward compatibility
def call_fast_api(file_path: str, filename: str, db_type: str, db_config: dict = None, 
                  use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                  process_large_files: bool = True, use_turbo: bool = False, batch_size: int = 256):
    """Legacy wrapper for Fast API call"""
    client = APIClient()
    return client.call_fast_api(file_path, filename, db_type, db_config, use_openai, openai_api_key, 
                               openai_base_url, process_large_files, use_turbo, batch_size)

def call_config1_api(file_path: str, filename: str, config: dict, db_config: dict = None,
                    use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                    process_large_files: bool = True, use_turbo: bool = False, batch_size: int = 256):
    """Legacy wrapper for Config1 API call"""
    client = APIClient()
    return client.call_config1_api(file_path, filename, config, db_config, use_openai, openai_api_key,
                                  openai_base_url, process_large_files, use_turbo, batch_size)

def call_retrieve_api(query: str, k: int = 5):
    """Legacy wrapper for retrieve API call"""
    client = APIClient()
    return client.call_retrieve_api(query, k)

def get_system_info_api():
    """Legacy wrapper for system info API call"""
    client = APIClient()
    return client.get_system_info_api()

def get_file_info_api():
    """Legacy wrapper for file info API call"""
    client = APIClient()
    return client.get_file_info_api()

def get_capabilities_api():
    """Legacy wrapper for capabilities API call"""
    client = APIClient()
    return client.get_capabilities_api()

def db_test_connection_api(payload: dict):
    """Legacy wrapper for DB test connection API call"""
    client = APIClient()
    return client.db_test_connection_api(payload)

def db_list_tables_api(payload: dict):
    """Legacy wrapper for DB list tables API call"""
    client = APIClient()
    return client.db_list_tables_api(payload)