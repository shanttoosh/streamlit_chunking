# Validation Utilities
import pandas as pd
from typing import Dict, Any, List, Optional

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate dataframe and return validation results"""
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    # Check if dataframe is empty
    if df.empty:
        validation_results["is_valid"] = False
        validation_results["errors"].append("DataFrame is empty")
        return validation_results
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        validation_results["warnings"].append("Duplicate column names found")
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        validation_results["warnings"].append(f"Empty columns found: {empty_columns}")
    
    # Check for columns with all same values
    constant_columns = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_columns.append(col)
    if constant_columns:
        validation_results["warnings"].append(f"Constant columns found: {constant_columns}")
    
    # Add basic info
    validation_results["info"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "dtypes": df.dtypes.to_dict()
    }
    
    return validation_results

def validate_chunking_params(chunk_size: int, overlap: int) -> Dict[str, Any]:
    """Validate chunking parameters"""
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    if chunk_size <= 0:
        validation_results["is_valid"] = False
        validation_results["errors"].append("Chunk size must be positive")
    
    if overlap < 0:
        validation_results["is_valid"] = False
        validation_results["errors"].append("Overlap cannot be negative")
    
    if overlap >= chunk_size:
        validation_results["warnings"].append("Overlap is greater than or equal to chunk size")
    
    if chunk_size > 10000:
        validation_results["warnings"].append("Very large chunk size may cause memory issues")
    
    return validation_results

def validate_database_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate database configuration"""
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    db_type = config.get("db_type", "").lower()
    
    if db_type not in ["mysql", "postgresql", "sqlite"]:
        validation_results["is_valid"] = False
        validation_results["errors"].append("Invalid database type")
        return validation_results
    
    if db_type in ["mysql", "postgresql"]:
        required_fields = ["host", "port", "username", "password", "database"]
        for field in required_fields:
            if not config.get(field):
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Missing required field: {field}")
    
    return validation_results

def validate_model_choice(model_choice: str) -> Dict[str, Any]:
    """Validate embedding model choice"""
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    valid_models = [
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L6-v2",
        "text-embedding-ada-002"
    ]
    
    if model_choice not in valid_models:
        validation_results["warnings"].append(f"Unknown model: {model_choice}")
    
    return validation_results

def validate_storage_choice(storage_choice: str) -> Dict[str, Any]:
    """Validate storage choice"""
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    valid_storages = ["faiss", "chroma"]
    
    if storage_choice not in valid_storages:
        validation_results["is_valid"] = False
        validation_results["errors"].append(f"Invalid storage choice: {storage_choice}")
    
    return validation_results

def validate_file_upload(file) -> Dict[str, Any]:
    """Validate uploaded file"""
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    if not file:
        validation_results["is_valid"] = False
        validation_results["errors"].append("No file provided")
        return validation_results
    
    # Check file size (assuming file has size attribute)
    if hasattr(file, 'size') and file.size > 100 * 1024 * 1024:  # 100MB
        validation_results["warnings"].append("Large file detected, processing may take time")
    
    # Check file extension
    if hasattr(file, 'filename'):
        filename = file.filename.lower()
        if not filename.endswith('.csv'):
            validation_results["warnings"].append("File is not a CSV file")
    
    return validation_results
