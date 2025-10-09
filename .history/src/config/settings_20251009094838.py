# Configuration Settings
import os
from typing import Optional

class Settings:
    """Application settings and configuration"""
    
    # API Settings
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    UI_PORT: int = 8501
    
    # Processing Settings
    LARGE_FILE_THRESHOLD: int = 10 * 1024 * 1024  # 10MB
    MAX_MEMORY_USAGE: float = 0.8  # 80% of available memory
    BATCH_SIZE: int = 2000
    EMBEDDING_BATCH_SIZE: int = 256
    PARALLEL_WORKERS: int = 6
    
    # Storage Settings
    CHROMADB_PATH: str = "storage/chromadb"
    FAISS_PATH: str = "storage/faiss"
    CACHE_PATH: str = "storage/cache"
    STATE_FILE: str = "current_state.pkl"
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
    
    # Default Models
    DEFAULT_FAST_MODEL: str = "paraphrase-MiniLM-L6-v2"
    DEFAULT_CONFIG1_MODEL: str = "paraphrase-MiniLM-L6-v2"
    DEFAULT_DEEP_MODEL: str = "paraphrase-MiniLM-L6-v2"
    
    # Chunking Defaults
    DEFAULT_CHUNK_SIZE: int = 400
    DEFAULT_OVERLAP: int = 50
    DEFAULT_TOKEN_LIMIT: int = 2000
    DEFAULT_N_CLUSTERS: int = 10
    
    # Storage Defaults
    DEFAULT_STORAGE: str = "faiss"
    DEFAULT_RETRIEVAL_METRIC: str = "cosine"
    
    # Database Settings
    MYSQL_DEFAULT_PORT: int = 3306
    POSTGRESQL_DEFAULT_PORT: int = 5432

# Global settings instance
settings = Settings()
