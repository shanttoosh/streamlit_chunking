# Database Connection and Import Module
import pandas as pd
import numpy as np
import time
import logging
import gc
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Performance Configuration
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10MB
BATCH_SIZE = 2000

def connect_mysql(host: str, port: int, username: str, password: str, database: str):
    """Connect to MySQL database"""
    try:
        import mysql.connector
        return mysql.connector.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            database=database
        )
    except ImportError:
        raise ImportError("mysql-connector-python is required for MySQL support. Please install 'mysql-connector-python'.")

def connect_postgresql(host: str, port: int, username: str, password: str, database: str):
    """Connect to PostgreSQL database"""
    try:
        import psycopg2
        return psycopg2.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            dbname=database
        )
    except ImportError:
        raise ImportError("psycopg2 is required for PostgreSQL support. Please install 'psycopg2-binary'.")

def get_table_list(conn, db_type: str):
    """Dynamic table discovery with fallback queries for MySQL and PostgreSQL"""
    fallback_queries = {
        "mysql": [
            "SHOW TABLES",
            "SELECT table_name FROM information_schema.tables WHERE table_schema=DATABASE()",
            "SELECT table_name FROM information_schema.tables"
        ],
        "postgresql": [
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'",
            "SELECT tablename FROM pg_tables WHERE schemaname='public'",
            "SELECT table_name FROM information_schema.tables"
        ]
    }
    
    queries = fallback_queries.get(db_type, [])
    
    for query in queries:
        try:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            cur.close()
            if rows:
                return [r[0] for r in rows]
        except Exception as e:
            logger.warning(f"Query failed: {query}, Error: {e}")
            continue
    
    return []

def import_table_to_dataframe(conn, table_name: str) -> pd.DataFrame:
    """Import table to DataFrame"""
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    return df

def get_table_size(conn, table_name: str) -> int:
    """Estimate table size in bytes"""
    try:
        cursor = conn.cursor()
        if hasattr(conn, 'cursor'):  # PostgreSQL
            cursor.execute(f"SELECT pg_relation_size('{table_name}')")
            size = cursor.fetchone()[0]
        else:  # MySQL
            cursor.execute(f"SELECT data_length + index_length FROM information_schema.tables WHERE table_name = '{table_name}'")
            size = cursor.fetchone()[0] or 0
        cursor.close()
        return size
    except:
        return 0  # Return 0 if cannot determine size

def import_large_table_to_dataframe(conn, table_name: str, chunk_size: int = 20000):
    """Import large tables in chunks to avoid memory issues"""
    chunks = []
    offset = 0
    
    while True:
        query = f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset}"
        try:
            chunk_df = pd.read_sql(query, conn)
        except:
            # Fallback to regular import
            query = f"SELECT * FROM {table_name}"
            chunk_df = pd.read_sql(query, conn)
            chunks.append(chunk_df)
            break
        
        if chunk_df.empty:
            break
            
        chunks.append(chunk_df)
        offset += chunk_size
        
        # Clear memory by consolidating chunks periodically
        if len(chunks) * chunk_size > 100000:  # Process every 100k rows
            partial_df = pd.concat(chunks, ignore_index=True)
            chunks = [partial_df]
            gc.collect()
    
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def is_large_table(conn, table_name: str, threshold_mb: int = 100) -> bool:
    """Check if database table is considered large"""
    try:
        table_size = get_table_size(conn, table_name)
        return table_size > threshold_mb * 1024 * 1024
    except:
        return False  # If can't determine size, assume it's not large

def can_load_file(file_size: int) -> bool:
    """Check if file can be loaded into memory"""
    try:
        import psutil
        available_memory = psutil.virtual_memory().available
        return file_size < available_memory * 0.8  # Use max 80% of available memory
    except:
        return file_size < LARGE_FILE_THRESHOLD  # Fallback to threshold

def get_available_memory():
    """Get available system memory in bytes"""
    try:
        import psutil
        return psutil.virtual_memory().available
    except:
        return LARGE_FILE_THRESHOLD * 10  # Fallback

def process_large_file_in_batches(file_path: str, processing_callback, batch_size: int = BATCH_SIZE):
    """Process large files in batches"""
    try:
        # Read file in chunks
        chunk_iter = pd.read_csv(file_path, chunksize=batch_size)
        results = []
        
        for chunk in chunk_iter:
            result = processing_callback(chunk)
            results.append(result)
            
            # Clear memory
            del chunk
            gc.collect()
        
        return results
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return []

def get_file_hash(file_path):
    """Generate hash for file caching"""
    import hashlib
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_cached_result(file_hash, processing_mode, config_hash):
    """Get cached processing result"""
    import pickle
    import os
    
    cache_file = f"processing_cache/{file_hash}_{processing_mode}_{config_hash}.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def cache_result(file_hash, processing_mode, config_hash, result):
    """Cache processing result"""
    import pickle
    import os
    
    os.makedirs("processing_cache", exist_ok=True)
    cache_file = f"processing_cache/{file_hash}_{processing_mode}_{config_hash}.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        logger.warning(f"Failed to cache result: {e}")

def process_large_file(file_path: str, processing_mode: str = "fast", **kwargs):
    """Process large files with appropriate strategy"""
    file_size = os.path.getsize(file_path)
    
    if can_load_file(file_size):
        # Load entire file
        df = pd.read_csv(file_path)
        return process_file_direct(df, processing_mode, **kwargs)
    else:
        # Process in batches
        return process_large_file_in_batches(file_path, 
                                           lambda chunk: process_file_direct(chunk, processing_mode, **kwargs))

def process_file_direct(file_path: str, processing_mode: str = "fast", **kwargs):
    """Process file directly (placeholder for actual processing)"""
    # This would call the appropriate pipeline based on mode
    # For now, just return a placeholder
    return {"status": "processed", "mode": processing_mode, "file": file_path}

def test_database_connection(db_type: str, host: str, port: int, username: str, password: str, database: str):
    """Test database connection"""
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"success": False, "error": "Unsupported database type"}
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        
        return {"success": True, "message": f"Successfully connected to {db_type} database"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def list_database_tables(db_type: str, host: str, port: int, username: str, password: str, database: str):
    """List tables in database"""
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"success": False, "error": "Unsupported database type"}
        
        tables = get_table_list(conn, db_type)
        conn.close()
        
        return {"success": True, "tables": tables}
    except Exception as e:
        return {"success": False, "error": str(e)}

def import_database_table(db_type: str, host: str, port: int, username: str, password: str, database: str, table_name: str):
    """Import table from database"""
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"success": False, "error": "Unsupported database type"}
        
        # Check if table is large
        if is_large_table(conn, table_name):
            df = import_large_table_to_dataframe(conn, table_name)
        else:
            df = import_table_to_dataframe(conn, table_name)
        
        conn.close()
        
        return {"success": True, "dataframe": df, "rows": len(df), "columns": len(df.columns)}
    except Exception as e:
        return {"success": False, "error": str(e)}