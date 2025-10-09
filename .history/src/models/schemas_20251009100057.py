# Pydantic Schemas for API
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

# Base Models
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: Optional[str] = None

# Processing Models
class ProcessingRequest(BaseModel):
    """Processing request model"""
    file: Optional[str] = None
    chunk_method: str = "recursive"
    chunk_size: int = 400
    overlap: int = 50
    document_key_column: Optional[str] = None
    token_limit: int = 2000
    retrieval_metric: str = "cosine"
    model_choice: str = "paraphrase-MiniLM-L6-v2"
    storage_choice: str = "faiss"
    apply_default_preprocessing: bool = True
    use_openai: bool = False
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    process_large_files: bool = True
    use_turbo: bool = False
    batch_size: int = 256

class ProcessingResponse(BaseModel):
    """Processing response model"""
    rows: int
    chunks: int
    stored: str
    embedding_model: str
    retrieval_ready: bool
    turbo_mode: Optional[bool] = None
    processing_time: Optional[float] = None

# Database Models
class DatabaseConfig(BaseModel):
    """Database configuration model"""
    db_type: str = "sqlite"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    table_name: Optional[str] = None

class DatabaseTestResponse(BaseModel):
    """Database test response model"""
    connected: bool
    message: str
    tables: Optional[List[str]] = None

# Retrieval Models
class RetrievalRequest(BaseModel):
    """Retrieval request model"""
    query: str
    k: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None

class RetrievalResult(BaseModel):
    """Single retrieval result"""
    rank: int
    content: str
    similarity: float
    distance: float
    metadata: Optional[Dict[str, Any]] = None

class RetrievalResponse(BaseModel):
    """Retrieval response model"""
    query: str
    k: int
    results: List[RetrievalResult]

# Export Models
class ExportRequest(BaseModel):
    """Export request model"""
    export_type: str = "chunks"  # chunks, embeddings, embeddings_text, preprocessed, deep_chunks, deep_embeddings

# System Models
class SystemInfo(BaseModel):
    """System information model"""
    memory_usage: str
    available_memory: str
    total_memory: str
    large_file_support: bool
    max_recommended_file_size: str
    embedding_batch_size: int
    parallel_workers: int

class FileInfo(BaseModel):
    """File information model"""
    filename: Optional[str] = None
    size: Optional[int] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    file_type: Optional[str] = None

# Deep Config Models
class PreprocessingConfig(BaseModel):
    """Preprocessing configuration"""
    fill_null_strategy: Optional[Dict[str, str]] = None
    type_conversions: Optional[Dict[str, str]] = None
    remove_stopwords_flag: bool = False

class ChunkingConfig(BaseModel):
    """Chunking configuration"""
    method: str = "fixed"
    chunk_size: int = 400
    overlap: int = 50
    key_column: Optional[str] = None
    token_limit: int = 2000
    preserve_headers: bool = True
    n_clusters: int = 10

class EmbeddingConfig(BaseModel):
    """Embedding configuration"""
    model_name: str = "paraphrase-MiniLM-L6-v2"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    batch_size: int = 64
    use_parallel: bool = True

class StorageConfig(BaseModel):
    """Storage configuration"""
    type: str = "chroma"
    collection_name: Optional[str] = None

class DeepConfigRequest(BaseModel):
    """Deep config request model"""
    preprocessing: Optional[PreprocessingConfig] = None
    chunking: Optional[ChunkingConfig] = None
    embedding: Optional[EmbeddingConfig] = None
    storage: Optional[StorageConfig] = None

# OpenAI Compatible Models
class OpenAIEmbeddingRequest(BaseModel):
    """OpenAI embedding request model"""
    model: str = "text-embedding-ada-002"
    input: Union[str, List[str]]
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

class OpenAIEmbeddingData(BaseModel):
    """OpenAI embedding data model"""
    object: str = "embedding"
    index: int
    embedding: List[float]

class OpenAIEmbeddingResponse(BaseModel):
    """OpenAI embedding response model"""
    object: str = "list"
    data: List[OpenAIEmbeddingData]
    model: str
    usage: Dict[str, int]
