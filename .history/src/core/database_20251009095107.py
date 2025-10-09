# Database Operations
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
import gc

logger = logging.getLogger(__name__)

def connect_mysql(host: str, port: int, username: str, password: str, database: str):
    """Connect to MySQL database"""
    import mysql.connector
    return mysql.connector.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database
    )

def connect_postgresql(host: str, port: int, username: str, password: str, database: str):
    """Connect to PostgreSQL database"""
    import psycopg2
    return psycopg2.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        dbname=database
    )

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
            print(f"Query failed: {query}, Error: {e}")
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

def test_database_connection(db_type: str, host: str, port: int, username: str, password: str, database: str) -> Dict[str, Any]:
    """Test database connection"""
    try:
        if db_type.lower() == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type.lower() == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"connected": False, "message": "Invalid database type"}
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        
        return {"connected": True, "message": "Connection successful"}
    except Exception as e:
        return {"connected": False, "message": str(e)}

def get_database_tables(db_type: str, host: str, port: int, username: str, password: str, database: str) -> Dict[str, Any]:
    """Get list of tables from database"""
    try:
        if db_type.lower() == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type.lower() == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"success": False, "message": "Invalid database type"}
        
        tables = get_table_list(conn, db_type)
        conn.close()
        
        return {"success": True, "tables": tables}
    except Exception as e:
        return {"success": False, "message": str(e)}
