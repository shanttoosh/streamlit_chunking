# Health Check Routes
from fastapi import APIRouter
from ...core.pipelines import get_file_info
from ...core.database import test_database_connection
from ...utils.file_utils import get_file_info as get_file_info_util
import psutil

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "message": "API is running"}

@router.get("/system_info")
async def get_system_info():
    """Get system information including memory usage"""
    memory = psutil.virtual_memory()
    return {
        "memory_usage": f"{memory.percent}%",
        "available_memory": f"{memory.available / (1024**3):.2f} GB",
        "total_memory": f"{memory.total / (1024**3):.2f} GB",
        "large_file_support": True,
        "max_recommended_file_size": "3GB+",
        "embedding_batch_size": 256,
        "parallel_workers": 6
    }

@router.get("/file_info")
async def get_file_info_endpoint():
    """Get stored file information"""
    file_info = get_file_info()
    return file_info or {}

@router.get("/capabilities")
async def get_capabilities():
    """Get system capabilities"""
    return {
        "large_file_support": True,
        "performance_features": {
            "turbo_mode": True,
            "parallel_processing": True,
            "batch_processing": True
        },
        "supported_formats": ["csv"],
        "supported_databases": ["mysql", "postgresql", "sqlite"],
        "supported_models": [
            "paraphrase-MiniLM-L6-v2",
            "all-MiniLM-L6-v2", 
            "text-embedding-ada-002"
        ],
        "supported_storage": ["faiss", "chroma"],
        "supported_chunking": ["fixed", "recursive", "semantic", "document"]
    }
