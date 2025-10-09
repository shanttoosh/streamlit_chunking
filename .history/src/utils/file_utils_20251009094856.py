# File Utilities
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union
import pandas as pd

def save_uploaded_file(uploaded_file, temp_dir: str = "data/temp") -> str:
    """Save uploaded file to temporary location"""
    # Create temp directory if it doesn't exist
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = int(time.time())
    filename = f"{timestamp}_{uploaded_file.filename}"
    file_path = os.path.join(temp_dir, filename)
    
    # Save file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(uploaded_file.file, f)
    
    return file_path

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(file_path)

def get_file_info(file_path: str) -> dict:
    """Get comprehensive file information"""
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    file_size = get_file_size(file_path)
    
    # Try to get CSV info
    try:
        df = pd.read_csv(file_path, nrows=1)
        return {
            "filename": os.path.basename(file_path),
            "size": file_size,
            "file_type": "csv",
            "columns": len(df.columns),
            "column_names": df.columns.tolist()
        }
    except Exception:
        return {
            "filename": os.path.basename(file_path),
            "size": file_size,
            "file_type": "unknown"
        }

def cleanup_temp_files(file_path: str):
    """Clean up temporary files"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {e}")

def can_load_file(file_size: int, max_memory_usage: float = 0.8) -> bool:
    """Check if file can be safely loaded into memory"""
    try:
        import psutil
        available_memory = psutil.virtual_memory().available
        return file_size < available_memory * max_memory_usage
    except ImportError:
        # Fallback if psutil is not available
        return file_size < 1024 * 1024 * 1024  # 1GB fallback

def get_file_hash(file_path: str) -> str:
    """Generate hash for file caching"""
    import hashlib
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def ensure_directory_exists(directory: str):
    """Ensure directory exists, create if not"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def move_file_to_storage(source_path: str, destination_dir: str, filename: Optional[str] = None) -> str:
    """Move file to storage directory"""
    ensure_directory_exists(destination_dir)
    
    if filename is None:
        filename = os.path.basename(source_path)
    
    destination_path = os.path.join(destination_dir, filename)
    shutil.move(source_path, destination_path)
    return destination_path
