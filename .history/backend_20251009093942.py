# backend.py - COMPLETE UPDATED VERSION
import pandas as pd
import numpy as np
import re
import time
import os
import psutil
import shutil
from typing import List, Dict, Any, Optional, Tuple, Union
import chromadb
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
import gc
from pathlib import Path
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Global variables to store current state for retrieval
current_model = None
current_store_info = None
current_chunks = None
current_embeddings = None
current_metadata = None
current_df = None
current_file_info = None

# Persistent state storage for web API
STATE_FILE = "current_state.pkl"

def save_state():
    """Save current state to disk for persistence across API calls"""
    try:
        state = {
            'current_model': current_model,
            'current_store_info': current_store_info,
            'current_chunks': current_chunks,
            'current_embeddings': current_embeddings,
            'current_df': current_df,
            'current_file_info': current_file_info
        }
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(state, f)
        logger.info("State saved to disk")
    except Exception as e:
        logger.error(f"Failed to save state: {e}")

def load_state():
    """Load state from disk"""
    global current_model, current_store_info, current_chunks, current_embeddings, current_df, current_file_info
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'rb') as f:
                state = pickle.load(f)
            current_model = state.get('current_model')
            current_store_info = state.get('current_store_info')
            current_chunks = state.get('current_chunks')
            current_embeddings = state.get('current_embeddings')
            current_df = state.get('current_df')
            current_file_info = state.get('current_file_info')
            logger.info("State loaded from disk")
            return True
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
    return False

# -----------------------------
# 🔹 Performance Optimization Configuration
# -----------------------------
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10MB
MAX_MEMORY_USAGE = 0.8  # 80% of available memory
BATCH_SIZE = 2000  # Increased for 3GB files
EMBEDDING_BATCH_SIZE = 256  # Increased for 3GB files
PARALLEL_WORKERS = 6  # Increased for 3GB files
CACHE_DIR = "processing_cache"

# -----------------------------
# 🔹 Performance Optimized Models
# -----------------------------
FAST_MODELS = {
    "paraphrase-MiniLM-L6-v2": {"speed": "fastest", "quality": "good"},
    "all-MiniLM-L6-v2": {"speed": "fast", "quality": "good"},
    "text-embedding-ada-002": {"speed": "medium", "quality": "excellent"}
}

# -----------------------------
# 🔹 Cache System for Performance
# -----------------------------
def get_file_hash(file_path):
    """Generate hash for file caching"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def get_cached_result(file_hash, processing_mode, config_hash):
    """Get cached processing results"""
    cache_file = os.path.join(CACHE_DIR, f"{file_hash}_{processing_mode}_{config_hash}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                logger.info("Loading from cache for faster processing")
                return pickle.load(f)
        except:
            pass
    return None

def cache_result(file_hash, processing_mode, config_hash, result):
    """Cache processing results"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{file_hash}_{processing_mode}_{config_hash}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except Exception as e:
        logger.warning(f"Could not cache result: {e}")

# -----------------------------
# 🔹 Large File Handling Configuration (3GB Support)
# -----------------------------
def get_available_memory():
    """Get available system memory in bytes"""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        # Fallback if psutil is not available
        import os
        if os.name == 'nt':  # Windows
            return 8 * 1024 * 1024 * 1024  # Assume 8GB available
        else:  # Unix-like
            return 4 * 1024 * 1024 * 1024  # Assume 4GB available

def can_load_file(file_size: int) -> bool:
    """Check if file can be safely loaded into memory"""
    available_memory = get_available_memory()
    return file_size < available_memory * MAX_MEMORY_USAGE

def process_large_file_in_batches(file_path: str, processing_callback, batch_size: int = BATCH_SIZE):
    """
    Process large CSV file in batches to avoid memory issues
    """
    chunks_processed = []
    total_rows = 0
    
    # First, get total rows for progress tracking
    with open(file_path, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # Subtract header
    
    # Process in batches
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        processed_chunk = processing_callback(chunk)
        chunks_processed.extend(processed_chunk)
        gc.collect()  # Force garbage collection
    
    return chunks_processed

# -----------------------------
# 🔹 Database Size Estimation & Large Table Handling
# -----------------------------
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

# 🔹 OpenAI API Standards
# -----------------------------
class OpenAIEmbeddingAPI:
    """OpenAI-compatible embedding API"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None, base_url: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.is_local = not api_key  # If no API key, use local model
    
    def encode(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Encode texts using OpenAI API or local fallback"""
        if self.is_local:
            # Use local model as fallback
            from sentence_transformers import SentenceTransformer
            local_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            embeddings = local_model.encode(texts, batch_size=batch_size)
            return np.array(embeddings).astype("float32")
        else:
            # Use OpenAI API
            import openai
            openai.api_key = self.api_key
            if self.base_url:
                openai.base_url = self.base_url
            
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                try:
                    response = openai.embeddings.create(
                        model=self.model_name,
                        input=batch_texts
                    )
                    batch_embeddings = [data.embedding for data in response.data]
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"OpenAI API error: {e}")
                    # Fallback to local model
                    from sentence_transformers import SentenceTransformer
                    local_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                    fallback_embeddings = local_model.encode(batch_texts)
                    embeddings.extend(fallback_embeddings)
            
            return np.array(embeddings).astype("float32")

# -----------------------------
# 🔹 Database Connection Helpers (single-table import)
# -----------------------------
def connect_mysql(host: str, port: int, username: str, password: str, database: str):
    import mysql.connector
    return mysql.connector.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database
    )

def connect_postgresql(host: str, port: int, username: str, password: str, database: str):
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
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    return df

current_sql_conn = None

# 🔹 NEW: Column Data Type Conversion
# -----------------------------
def convert_column_types(df: pd.DataFrame, column_types: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Convert column data types based on user specification
    """
    start_time = time.time()
    df_converted = df.copy()
    
    conversion_results = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    for column, target_type in column_types.items():
        if column not in df_converted.columns:
            conversion_results['skipped'].append(f"Column '{column}' not found")
            continue
            
        try:
            if target_type == 'string':
                df_converted[column] = df_converted[column].astype(str)
                conversion_results['successful'].append(f"{column} -> string")
                
            elif target_type == 'numeric':
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                conversion_results['successful'].append(f"{column} -> numeric")
                
            elif target_type == 'integer':
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce').fillna(0).astype(int)
                conversion_results['successful'].append(f"{column} -> integer")
                
            elif target_type == 'float':
                df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce').astype(float)
                conversion_results['successful'].append(f"{column} -> float")
                
            elif target_type == 'datetime':
                df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
                conversion_results['successful'].append(f"{column} -> datetime")
                
            elif target_type == 'boolean':
                # Try to convert common boolean representations
                if df_converted[column].dtype == 'object':
                    true_values = ['true', 'yes', '1', 't', 'y']
                    false_values = ['false', 'no', '0', 'f', 'n']
                    df_converted[column] = df_converted[column].astype(str).str.lower().isin(true_values)
                conversion_results['successful'].append(f"{column} -> boolean")
                
            elif target_type == 'category':
                df_converted[column] = df_converted[column].astype('category')
                conversion_results['successful'].append(f"{column} -> category")
                
            else:
                conversion_results['skipped'].append(f"Unknown type '{target_type}' for column '{column}'")
                
        except Exception as e:
            conversion_results['failed'].append(f"{column} -> {target_type}: {str(e)}")
    
    logger.info(f"Column type conversion completed in {time.time() - start_time:.2f}s")
    logger.info(f"Conversion results: {len(conversion_results['successful'])} successful, "
                f"{len(conversion_results['failed'])} failed, "
                f"{len(conversion_results['skipped'])} skipped")
    
    return df_converted, conversion_results

# -----------------------------
# 🔹 Enhanced Text Cleaning Functions
# -----------------------------
def clean_text_advanced(text_series: pd.Series, lowercase: bool = True, remove_delimiters: bool = True, 
                       remove_whitespace: bool = True) -> pd.Series:
    """
    Advanced text cleaning for string columns
    """
    cleaned_series = text_series.astype(str)
    
    if lowercase:
        cleaned_series = cleaned_series.str.lower()
    
    if remove_delimiters:
        # Remove common delimiters and special characters
        cleaned_series = cleaned_series.str.replace(r'[^\w\s]', ' ', regex=True)
    
    if remove_whitespace:
        # Remove extra whitespace
        cleaned_series = cleaned_series.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return cleaned_series

# -----------------------------
# 🔹 Token Count Estimation
# -----------------------------
def estimate_token_count(text: str) -> int:
    """
    Improved token count estimation
    Uses better heuristics for token counting
    """
    # More accurate token estimation: words + punctuation
    words = len(text.split())
    punctuation = len([c for c in text if c in '.,!?;:()[]{}"\''])
    
    # Average of 1.3 tokens per word + punctuation as separate tokens
    return int(words * 1.3 + punctuation)

# -----------------------------
# 🔹 Preprocessing with Large File Support
# -----------------------------
def preprocess_basic(df: pd.DataFrame, null_handling="keep", fill_value=None):
    start_time = time.time()
    if null_handling == "drop":
        df = df.dropna().reset_index(drop=True)
    elif null_handling == "fill" and fill_value is not None:
        df = df.fillna(fill_value)
    logger.info(f"Basic preprocessing completed in {time.time() - start_time:.2f}s")
    return df

def preprocess_auto_fast(df: pd.DataFrame):
    """
    Auto preprocessing for Fast Mode: lowercase + remove delimiters + remove whitespace
    """
    start_time = time.time()
    
    # Handle nulls by dropping
    df = df.dropna().reset_index(drop=True)
    
    # Apply text cleaning to all string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = clean_text_advanced(df[col], lowercase=True, remove_delimiters=True, remove_whitespace=True)
    
    logger.info(f"Auto Fast preprocessing completed in {time.time() - start_time:.2f}s")
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize column names"""
    new_columns = []
    for i, col in enumerate(df.columns):
        if pd.isna(col) or str(col).strip() == "":
            new_col = f"column_{i+1}"
        else:
            # Clean column name: lowercase, replace spaces/special chars with underscores
            new_col = str(col).strip().lower()
            new_col = re.sub(r'[^a-z0-9_]', '_', new_col)
            new_col = re.sub(r'_+', '_', new_col)  # Replace multiple underscores with single
            new_col = new_col.strip('_')  # Remove leading/trailing underscores
            if not new_col or new_col.startswith('_'):
                new_col = f"column_{i+1}"
        new_columns.append(new_col)
    df.columns = new_columns
    return df

def preprocess_optimized_fast(df: pd.DataFrame):
    """
    OPTIMIZED: Faster preprocessing for large files - minimal operations
    """
    start_time = time.time()
    
    # Clean column names
    df = clean_column_names(df)
    
    # Only essential operations - drop nulls
    df = df.dropna().reset_index(drop=True)
    
    # Fast text cleaning - only lowercase, skip other operations for speed
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.lower()
    
    logger.info(f"Optimized fast preprocessing completed in {time.time() - start_time:.2f}s")
    return df

def preprocess_auto_config1(df: pd.DataFrame, null_handling="keep", fill_value=None):
    """
    Auto preprocessing for Config1 Mode: lowercase + remove delimiters + remove whitespace + null handling
    """
    start_time = time.time()
    
    # Handle nulls based on user choice
    if null_handling == "drop":
        df = df.dropna().reset_index(drop=True)
    elif null_handling == "fill" and fill_value is not None:
        df = df.fillna(fill_value)
    
    # Apply text cleaning to all string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = clean_text_advanced(df[col], lowercase=True, remove_delimiters=True, remove_whitespace=True)
    
    logger.info(f"Auto Config1 preprocessing completed in {time.time() - start_time:.2f}s")
    return df


# 🔹 Improved Fixed Chunking with Large File Support
# -----------------------------
def chunk_fixed(df: pd.DataFrame, chunk_size=400, overlap=50):
    """
    Improved fixed chunking with better text splitting
    """
    start_time = time.time()
    
    # Convert dataframe to text rows
    rows = []
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        rows.append(row_text)
    
    # Use improved text splitter with better parameters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Better separators
    )
    
    # Split the text
    text = "\n".join(rows)
    chunks = splitter.split_text(text)
    
    logger.info(f"Fixed chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks

# -----------------------------
# 🔹 Improved Recursive Key-Value Chunking
# -----------------------------
def chunk_recursive_keyvalue(df: pd.DataFrame, chunk_size=400, overlap=50):
    """
    Improved recursive key-value chunking with better handling
    """
    start_time = time.time()
    
    rows = []
    for _, row in df.iterrows():
        kv_pairs = [f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c])]
        rows.append(" | ".join(kv_pairs))
    
    big_text = "\n".join(rows)
    
    # Use improved text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " | ", " ", ""]  # Better separators for key-value format
    )
    
    chunks = splitter.split_text(big_text)
    logger.info(f"Recursive KV chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks

def chunk_semantic_cluster(df: pd.DataFrame, n_clusters=10):
    """Group rows into clusters based on semantic embeddings of rows."""
    start_time = time.time()
    sentences = df.astype(str).agg(" ".join, axis=1).tolist()
    
    # Use local model for semantic clustering
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    
    # Process in batches for large files
    if len(sentences) > 1000:
        embs = []
        for i in range(0, len(sentences), 1000):
            batch = sentences[i:i+1000]
            batch_embs = model.encode(batch)
            embs.extend(batch_embs)
        embs = np.array(embs)
    else:
        embs = model.encode(sentences)
        
    kmeans = KMeans(n_clusters=min(n_clusters, len(sentences)), random_state=42)
    labels = kmeans.fit_predict(embs)

    grouped = {}
    for sent, lab in zip(sentences, labels):
        grouped.setdefault(lab, []).append(sent)

    chunks = [" ".join(v) for v in grouped.values()]
    logger.info(f"Semantic clustering completed in {time.time() - start_time:.2f}s, created {len(chunks)} clusters")
    return chunks

# -----------------------------
# 🔹 NEW IMPROVED DOCUMENT-BASED CHUNKING
# -----------------------------
def document_based_chunking(df: pd.DataFrame, key_column: str, 
                          token_limit: int = 2000, 
                          preserve_headers: bool = True) -> Tuple[List[str], List[dict]]:
    """
    NEW IMPROVED: Document-based chunking with better token counting and grouping
    Groups rows by specified column and creates chunks based on token limits
    """
    start_time = time.time()
    
    if key_column not in df.columns:
        raise ValueError(f"Key column '{key_column}' not found in dataframe")
    
    chunks = []
    metas = []
    
    # Group by key column
    grouped = df.groupby(key_column)
    
    chunk_index = 0
    for key_value, group in grouped:
        # Convert entire group to text representation
        if preserve_headers:
            headers = " | ".join(group.columns.astype(str))
            rows_text = []
            for _, row in group.iterrows():
                row_text = " | ".join([f"{col}:{val}" for col, val in row.items() if pd.notna(val)])
                rows_text.append(row_text)
            group_text = f"HEADERS: {headers}\n" + "\n".join(rows_text)
        else:
            rows_text = []
            for _, row in group.iterrows():
                row_text = " | ".join([f"{col}:{val}" for col, val in row.items() if pd.notna(val)])
                rows_text.append(row_text)
            group_text = "\n".join(rows_text)
        
        # Improved token count estimation
        token_count = estimate_token_count(group_text)
        
        # If group fits within token limit, use as single chunk
        if token_count <= token_limit:
            chunks.append(group_text)
            metas.append({
                'chunk_index': chunk_index,
                'key_column': key_column,
                'key_value': str(key_value),
                'chunking_method': 'document_based',
                'token_count': token_count,
                'token_limit': token_limit,
                'group_size': len(group),
                'is_subchunk': False
            })
            chunk_index += 1
        else:
            # Improved sub-chunking: calculate optimal rows per chunk
            avg_tokens_per_row = token_count / len(group)
            rows_per_chunk = max(1, min(len(group), int(token_limit / avg_tokens_per_row)))
            
            # Ensure we don't create too many tiny chunks
            if len(group) / rows_per_chunk > 10:  # If we'd create more than 10 chunks
                rows_per_chunk = max(rows_per_chunk, len(group) // 10)
            
            for i in range(0, len(group), rows_per_chunk):
                end_idx = min(i + rows_per_chunk, len(group))
                sub_group = group.iloc[i:end_idx]
                
                # Convert sub-group to text
                if preserve_headers:
                    headers = " | ".join(sub_group.columns.astype(str))
                    rows_text = []
                    for _, row in sub_group.iterrows():
                        row_text = " | ".join([f"{col}:{val}" for col, val in row.items() if pd.notna(val)])
                        rows_text.append(row_text)
                    sub_text = f"HEADERS: {headers}\n" + "\n".join(rows_text)
                else:
                    rows_text = []
                    for _, row in sub_group.iterrows():
                        row_text = " | ".join([f"{col}:{val}" for col, val in row.items() if pd.notna(val)])
                        rows_text.append(row_text)
                    sub_text = "\n".join(rows_text)
                
                sub_token_count = estimate_token_count(sub_text)
                
                chunks.append(sub_text)
                metas.append({
                    'chunk_index': chunk_index,
                    'key_column': key_column,
                    'key_value': str(key_value),
                    'chunking_method': 'document_based',
                    'token_count': sub_token_count,
                    'token_limit': token_limit,
                    'group_size': len(group),
                    'subchunk_index': (i // rows_per_chunk) + 1,
                    'total_subchunks': (len(group) + rows_per_chunk - 1) // rows_per_chunk,
                    'is_subchunk': True,
                    'rows_in_chunk': len(sub_group)
                })
                chunk_index += 1
    
    logger.info(f"Document-based chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks, metas

# -----------------------------
# 🔹 Performance Optimized Embedding + Storage
# -----------------------------
def parallel_embed_texts(chunks, model_name="paraphrase-MiniLM-L6-v2", batch_size=EMBEDDING_BATCH_SIZE, num_workers=PARALLEL_WORKERS):
    """Parallel embedding for faster processing"""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    
    def embed_batch(batch_chunks):
        return model.encode(batch_chunks, batch_size=batch_size)
    
    # Split chunks into batches for parallel processing
    chunk_batches = [chunks[i:i + len(chunks)//num_workers] for i in range(0, len(chunks), len(chunks)//num_workers)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(embed_batch, chunk_batches))
    
    # Combine results
    embeddings = np.vstack(results)
    return model, embeddings

def embed_texts(chunks, model_name="paraphrase-MiniLM-L6-v2", openai_api_key=None, openai_base_url=None, batch_size=EMBEDDING_BATCH_SIZE, use_parallel=True):
    start_time = time.time()
    
    # Use parallel processing for large files when enabled
    if use_parallel and len(chunks) > 500 and not openai_api_key and "text-embedding" not in model_name.lower():
        logger.info(f"Using parallel processing for {len(chunks)} chunks")
        model, embeddings = parallel_embed_texts(chunks, model_name, batch_size)
        logger.info(f"Parallel embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
        return model, embeddings
    
    if openai_api_key or "text-embedding" in model_name.lower():
        # Use OpenAI API
        openai_model = OpenAIEmbeddingAPI(
            model_name=model_name if "text-embedding" in model_name.lower() else "text-embedding-ada-002",
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        embeddings = openai_model.encode(chunks, batch_size=batch_size)
        model = openai_model
        logger.info(f"OpenAI embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
    else:
        # Use local model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, batch_size=batch_size)
        logger.info(f"Local embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
    
    return model, np.array(embeddings).astype("float32")

def store_chroma(chunks, embeddings, collection_name="chunks_collection"):
    start_time = time.time()
    client = chromadb.PersistentClient(path="chromadb_store")
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name)
    
    # Add in batches for large collections
    batch_size = 1000
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_ids = [str(j) for j in range(i, end_idx)]
        
        col.add(
            ids=batch_ids,
            documents=batch_chunks,
            embeddings=batch_embeddings.tolist()
        )
    
    logger.info(f"Chroma storage completed in {time.time() - start_time:.2f}s, stored {len(chunks)} vectors")
    return {"type": "chroma", "collection": col, "collection_name": collection_name}

def store_faiss(embeddings):
    start_time = time.time()
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    
    # Add in batches for large embeddings
    batch_size = 10000
    for i in range(0, embeddings.shape[0], batch_size):
        end_idx = min(i + batch_size, embeddings.shape[0])
        batch_embeddings = embeddings[i:end_idx]
        index.add(batch_embeddings)
    
    logger.info(f"FAISS storage completed in {time.time() - start_time:.2f}s, stored {embeddings.shape[0]} vectors")
    return {"type": "faiss", "index": index}

# 🔹 Retrieval Functions with OpenAI Support
# -----------------------------
def retrieve_similar(query: str, k: int = 5):
    """Retrieve similar chunks using the current stored embeddings"""
    global current_model, current_store_info, current_chunks, current_embeddings
    
    start_time = time.time()
    
    # Load state from disk if globals are empty
    if current_model is None or current_store_info is None:
        logger.info("Globals empty, attempting to load state from disk")
        load_state()
    
    # Debug: Check global variable states
    logger.info(f"Retrieve debug - current_model: {current_model is not None}, current_store_info: {current_store_info is not None}, current_chunks: {len(current_chunks) if current_chunks else 0}")
    
    if current_model is None or current_store_info is None:
        return {"error": "No model or store available. Run a pipeline first."}
    
    # Encode query
    if hasattr(current_model, 'encode'):
        # Local model
        query_embedding = current_model.encode([query])
    else:
        # OpenAI model
        query_embedding = current_model.encode([query])
    
    query_arr = np.array(query_embedding).astype("float32")
    
    results = []
    
    if current_store_info["type"] == "faiss":
        # Enhanced FAISS retrieval with metadata support
        index = current_store_info["index"]
        faiss_data = current_store_info.get("data", {})
        
        # Use enhanced query function
        results = query_faiss_with_metadata(index, faiss_data, query_arr, k)
    
    elif current_store_info["type"] == "chroma":
        # Chroma retrieval
        collection = current_store_info["collection"]
        chroma_results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=["documents", "distances"]
        )
        
        for i, (doc, distance) in enumerate(zip(
            chroma_results["documents"][0], 
            chroma_results["distances"][0]
        )):
            similarity = 1 / (1 + distance)
            results.append({
                "rank": i + 1,
                "content": doc,
                "similarity": float(similarity),
                "distance": float(distance)
            })
    
    else:
        # Fallback: cosine similarity with raw embeddings
        if current_embeddings is not None:
            similarities = cosine_similarity(query_arr, current_embeddings)[0]
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            for i, idx in enumerate(top_indices):
                results.append({
                    "rank": i + 1,
                    "content": current_chunks[idx],
                    "similarity": float(similarities[idx]),
                    "distance": float(1 - similarities[idx])
                })
    
    logger.info(f"Retrieval completed in {time.time() - start_time:.2f}s, found {len(results)} results")
    return {"query": query, "k": k, "results": results}

# -----------------------------
# 🔹 Export Functions
# -----------------------------
def export_chunks():
    """Export current chunks as CSV format"""
    global current_chunks
    if current_chunks:
        # Create DataFrame with all chunks
        df = pd.DataFrame({
            'chunk_id': range(1, len(current_chunks) + 1),
            'chunk_text': current_chunks
        })
        return df.to_csv(index=False)
    return ""

def export_embeddings():
    """Export current embeddings as numpy array"""
    global current_embeddings
    return current_embeddings

def export_embeddings_json():
    """Export current embeddings as JSON format"""
    global current_embeddings, current_chunks
    if current_embeddings is not None and current_chunks is not None:
        # Create JSON structure with all embeddings
        embeddings_data = {
            "total_chunks": len(current_chunks),
            "vector_dimension": current_embeddings.shape[1] if len(current_embeddings.shape) > 1 else 0,
            "embeddings": []
        }
        
        for i, (chunk, embedding) in enumerate(zip(current_chunks, current_embeddings)):
            embeddings_data["embeddings"].append({
                "chunk_id": i + 1,
                "chunk_text": chunk,
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            })
        
        return json.dumps(embeddings_data, indent=2)
    return "{}"

# -----------------------------
# 🔹 System Info
# -----------------------------
def get_system_info():
    """Get system information including memory usage"""
    memory = psutil.virtual_memory()
    return {
        "memory_usage": f"{memory.percent}%",
        "available_memory": f"{memory.available / (1024**3):.2f} GB",
        "total_memory": f"{memory.total / (1024**3):.2f} GB",
        "large_file_support": True,
        "max_recommended_file_size": "3GB+",
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "parallel_workers": PARALLEL_WORKERS
    }

# -----------------------------
# 🔹 File Info Management
# -----------------------------
def set_file_info(file_info: Dict):
    """Store file information"""
    global current_file_info
    current_file_info = file_info

def get_file_info():
    """Get stored file information"""
    global current_file_info
    return current_file_info or {}

# -----------------------------
# 🔹 Enhanced Pipelines with Large File Support
# -----------------------------
def run_fast_pipeline(df, db_type="sqlite", db_config=None, file_info=None, 
                     use_openai=False, openai_api_key=None, openai_base_url=None,
                     use_turbo=False, batch_size=EMBEDDING_BATCH_SIZE):
    global current_model, current_store_info, current_chunks, current_embeddings, current_df
    
    current_df = df.copy()
    set_file_info(file_info)
    
    # Apply default preprocessing (automatic for Fast Mode)
    logger.info("Applying default preprocessing for Fast Mode")
    df = _load_csv(df) if not isinstance(df, pd.DataFrame) else df
    df = validate_and_normalize_headers(df)
    
    # Normalize text columns with default settings
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in text_cols:
        df[col] = normalize_text_column(df[col], lowercase=True, strip=True, remove_html_flag=True)
    
    # Use optimized preprocessing for turbo mode
    if use_turbo:
        df1 = preprocess_optimized_fast(df)
    else:
        df1 = preprocess_auto_fast(df)
    
    # FAST MODE DEFAULTS: semantic clustering with paraphrase model
    # Ensure minimum clusters for small datasets
    n_clusters = max(1, min(20, len(df1)//10)) if len(df1) > 0 else 1
    chunks = chunk_semantic_cluster(df1, n_clusters=n_clusters)
    
    # Choose embedding method - default to paraphrase for speed
    model_choice = "text-embedding-ada-002" if use_openai else "paraphrase-MiniLM-L6-v2"
    model, embs = embed_texts(chunks, model_choice, openai_api_key, openai_base_url, batch_size=batch_size, use_parallel=use_turbo)
    store = store_faiss(embs)
    
    # Store for retrieval
    current_model = model
    current_store_info = store
    current_chunks = chunks
    current_embeddings = embs
    
    # Debug: Log that globals are set
    logger.info(f"Fast pipeline completed - stored {len(chunks)} chunks, model: {model_choice}, store: {store['type']}")
    
    # Save state to disk for persistence across API calls
    save_state()
    
    return {
        "rows": len(df1), 
        "chunks": len(chunks), 
        "stored": store["type"],
        "embedding_model": model_choice,
        "retrieval_ready": True,
        "turbo_mode": use_turbo
    }

def run_config1_pipeline(df, chunk_method,
                         chunk_size, overlap, model_choice, storage_choice, 
                         db_config=None, file_info=None, use_openai=False, 
                         openai_api_key=None, openai_base_url=None,
                         use_turbo=False, batch_size=EMBEDDING_BATCH_SIZE,
                         document_key_column: str = None, token_limit: int = 2000,
                         retrieval_metric: str = "cosine", apply_default_preprocessing: bool = True):
    global current_model, current_store_info, current_chunks, current_embeddings, current_df
    
    current_df = df.copy()
    set_file_info(file_info)
    
    # Apply default preprocessing if enabled
    if apply_default_preprocessing:
        logger.info("Applying default preprocessing for Config-1 Mode")
        df = _load_csv(df) if not isinstance(df, pd.DataFrame) else df
        df = validate_and_normalize_headers(df)
        
        # Normalize text columns with default settings
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in text_cols:
            df[col] = normalize_text_column(df[col], lowercase=True, strip=True, remove_html_flag=True)
    
    df1 = df.copy()

    if chunk_method == "fixed":
        chunks = chunk_fixed(df1, chunk_size, overlap)
    elif chunk_method == "recursive":
        chunks = chunk_recursive_keyvalue(df1, chunk_size, overlap)
    elif chunk_method == "semantic":
        chunks = chunk_semantic_cluster(df1)
    else:  # document
        # Use provided key column or default to first column; do not preserve headers
        key_column = (document_key_column if document_key_column else (df1.columns[0] if len(df1.columns) > 0 else "id"))
        chunks, _ = document_based_chunking(df1, key_column, token_limit=int(token_limit), preserve_headers=False)

    # Use OpenAI if specified and model choice is OpenAI model
    actual_openai = use_openai or "text-embedding" in model_choice.lower()
    model, embs = embed_texts(chunks, model_choice, openai_api_key if actual_openai else None, openai_base_url, batch_size=batch_size, use_parallel=use_turbo)
    
    if storage_choice == "faiss":
        # Metric handling for FAISS
        # - euclidean: IndexFlatL2
        # - dot: IndexFlatIP
        # - cosine: normalize vectors then IP (approx cosine)
        faiss_metric = (retrieval_metric or "cosine").lower()
        try:
            import faiss
            import numpy as np
            vecs = embs
            if faiss_metric == "cosine":
                # Normalize rows to unit length
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vecs = vecs / norms
                index = faiss.IndexFlatIP(vecs.shape[1])
            elif faiss_metric == "dot":
                index = faiss.IndexFlatIP(vecs.shape[1])
            else:  # euclidean
                index = faiss.IndexFlatL2(vecs.shape[1])
            index.add(vecs)
            store = {"type": "faiss", "index": index, "metric": faiss_metric}
        except Exception:
            # Fallback to existing storage if faiss not available
            store = store_faiss(embs)
            store["metric"] = "euclidean"
    else:
        # Chroma metric space: "cosine", "l2", or "ip"
        space = (retrieval_metric or "cosine").lower()
        if space == "euclidean":
            space = "l2"
        try:
            import chromadb
            client = chromadb.PersistentClient(path="chromadb_store")
            col_name = f"config1_{int(time.time())}"
            # delete if exists
            try:
                client.delete_collection(col_name)
            except Exception:
                pass
            col = client.create_collection(col_name, metadata={"hnsw:space": space})
            # batch add
            batch_size = 1000
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                col.add(
                    ids=[str(j) for j in range(i, end_idx)],
                    documents=chunks[i:end_idx],
                    embeddings=embs[i:end_idx].tolist()
                )
            store = {"type": "chroma", "collection": col, "collection_name": col_name, "space": space}
        except Exception:
            store = store_chroma(chunks, embs, f"config1_{int(time.time())}")
            store["space"] = "cosine"

    # Store for retrieval
    current_model = model
    current_store_info = store
    current_chunks = chunks
    current_embeddings = embs
    
    # Save state to disk for persistence across API calls
    save_state()
    
    return {
        "rows": len(df1), 
        "chunks": len(chunks), 
        "stored": store["type"],
        "embedding_model": model_choice,
        "retrieval_ready": True,
        "turbo_mode": use_turbo
    }


# 🔹 Large File Processing Function
# -----------------------------
def process_large_file(file_path: str, processing_mode: str = "fast", **kwargs):
    """
    Process large files by handling them in batches
    """
    file_size = os.path.getsize(file_path)
    
    if not can_load_file(file_size):
        logger.warning(f"File size {file_size/(1024**3):.2f}GB may exceed safe memory limits")
    
    # Process based on mode
    if processing_mode == "fast":
        return process_large_file_fast(file_path, **kwargs)
    elif processing_mode == "config1":
        return process_large_file_config1(file_path, **kwargs)
    elif processing_mode == "deep":
        return process_large_file_deep(file_path, **kwargs)
    else:
        raise ValueError(f"Unknown processing mode: {processing_mode}")

def process_large_file_fast(file_path: str, **kwargs):
    """Process large file in fast mode with batches"""
    def process_batch(batch_df):
        # Use optimized preprocessing for large files
        processed_df = preprocess_optimized_fast(batch_df)
        chunks = chunk_semantic_cluster(processed_df)  # Use semantic clustering for large files
        return chunks

# -----------------------------
# 🔹 NEW: Document-Based Chunking (Multi-Key)
# -----------------------------
 

# -----------------------------
# 🔹 NEW: Unified Chunking Dispatcher
# -----------------------------
 
    
    all_chunks = process_large_file_in_batches(file_path, process_batch)
    
    # Embed and store all chunks
    model_choice = kwargs.get('model_choice', 'paraphrase-MiniLM-L6-v2')
    use_openai = kwargs.get('use_openai', False)
    openai_api_key = kwargs.get('openai_api_key')
    openai_base_url = kwargs.get('openai_base_url')
    batch_size = kwargs.get('batch_size', EMBEDDING_BATCH_SIZE)
    use_turbo = kwargs.get('use_turbo', True)
    
    model, embs = embed_texts(all_chunks, model_choice, openai_api_key, openai_base_url, batch_size=batch_size, use_parallel=use_turbo)
    store = store_faiss(embs)
    
    # Store for retrieval
    global current_model, current_store_info, current_chunks, current_embeddings
    current_model = model
    current_store_info = store
    current_chunks = all_chunks
    current_embeddings = embs
    
    return {
        "rows": "Large file processed in batches", 
        "chunks": len(all_chunks), 
        "stored": store["type"],
        "embedding_model": model_choice,
        "retrieval_ready": True,
        "large_file_processed": True,
        "turbo_mode": use_turbo
    }

def process_large_file_config1(file_path: str, **kwargs):
    """Process large file in config1 mode with batches"""
    # Similar implementation as fast mode but with config1 parameters
    # For brevity, implementing basic version
    return process_large_file_fast(file_path, **kwargs)

def process_large_file_deep(file_path: str, **kwargs):
    """Process large file in deep mode with batches"""
    # Similar implementation as fast mode but with deep parameters  
    # For brevity, implementing basic version
    return process_large_file_fast(file_path, **kwargs)

# -----------------------------
# 🔹 NEW: Direct File Processing from Filesystem
# -----------------------------
def process_file_direct(file_path: str, processing_mode: str = "fast", **kwargs):
    """
    Process file directly from filesystem path (no memory loading)
    This is the main function called by the API for large files
    """
    file_size = os.path.getsize(file_path)
    
    logger.info(f"Processing file directly from filesystem: {file_path} ({file_size/(1024**3):.2f}GB)")
    
    if not can_load_file(file_size):
        logger.warning(f"File size {file_size/(1024**3):.2f}GB may exceed safe memory limits")
    
    # Process based on mode using the existing process_large_file functions
    if processing_mode == "fast":
        return process_large_file_fast(file_path, **kwargs)
    elif processing_mode == "config1":
        return process_large_file_config1(file_path, **kwargs)
    elif processing_mode == "deep":
        return process_large_file_deep(file_path, **kwargs)
    else:
        raise ValueError(f"Unknown processing mode: {processing_mode}")

# -----------------------------
# 🔹 NEW: Database Large Table Detection
# -----------------------------
def is_large_table(conn, table_name: str, threshold_mb: int = 100) -> bool:
    """Check if database table is considered large"""
    try:
        table_size = get_table_size(conn, table_name)
        return table_size > threshold_mb * 1024 * 1024
    except:
        return False  # If can't determine size, assume it's not large        


# =============================================================================
# 🔹 NEW DEEP CONFIG FUNCTIONS - Enhanced Preprocessing, Chunking, Embedding, Storage
# =============================================================================

# -----------------------------
# 🔹 Enhanced Preprocessing Functions
# -----------------------------
def validate_and_normalize_headers_enhanced(columns):
    """Enhanced header validation and normalization"""
    new_columns = []
    for i, col in enumerate(columns):
        if col is None or str(col).strip() == "":
            new_col = f"column_{i+1}"
        else:
            new_col = str(col).strip().lower()
        new_columns.append(new_col)
    return new_columns

def normalize_text_column_enhanced(s: pd.Series, lowercase=True, strip=True, remove_html_flag=True):
    """Enhanced text normalization with HTML removal"""
    s = s.fillna('')
    if remove_html_flag:
        s = s.map(lambda x: re.sub('<[^<]+?>', ' ', str(x)) if isinstance(x, str) else x)
    if lowercase:
        s = s.map(lambda x: x.lower() if isinstance(x, str) else x)
    if strip:
        s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    s = s.map(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)
    return s

# -----------------------------
# 🔹 Original Preprocessing Functions (Fixed)
# -----------------------------
def _load_csv(input_obj):
    """Load CSV with proper encoding detection"""
    if isinstance(input_obj, pd.DataFrame):
        return input_obj.copy()
    if hasattr(input_obj, "read"):
        try:
            return pd.read_csv(input_obj, engine="python")
        except Exception:
            input_obj.seek(0)
            return pd.read_csv(input_obj)
    if isinstance(input_obj, str):
        try:
            return pd.read_csv(input_obj, engine="python")
        except Exception:
            # Try with chardet for encoding detection
            try:
                import chardet
                with open(input_obj, "rb") as fh:
                    raw = fh.read()
                enc = chardet.detect(raw).get("encoding", "utf-8")
                return pd.read_csv(input_obj, encoding=enc, engine="python")
            except ImportError:
                # Fallback without chardet
                with open(input_obj, "rb") as fh:
                    raw = fh.read(100)
                text = raw.decode("utf-8", errors="replace")
                from io import StringIO
                return pd.read_csv(StringIO(text))
    raise ValueError("Unsupported input type for _load_csv")

def remove_html(text):
    """Remove HTML tags from text"""
    if not isinstance(text, str):
        return text
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(text, "lxml").get_text(separator=' ')
    except ImportError:
        # Fallback without BeautifulSoup
        return re.sub('<[^<]+?>', ' ', text)
    except:
        return re.sub('<[^<]+?>', ' ', text)

def validate_and_normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize column headers"""
    new_columns = []
    for i, col in enumerate(df.columns):
        if pd.isna(col) or str(col).strip() == "":
            new_col = f"column_{i+1}"
        else:
            new_col = str(col).strip().lower()
        new_columns.append(new_col)
    df.columns = new_columns
    return df

def normalize_text_column(s: pd.Series, lowercase=True, strip=True, remove_html_flag=True):
    """Normalize text column with HTML removal, lowercase, and strip"""
    s = s.fillna('')
    if remove_html_flag:
        s = s.map(remove_html)
    if lowercase:
        s = s.map(lambda x: x.lower() if isinstance(x, str) else x)
    if strip:
        s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    s = s.map(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)
    return s

def apply_type_conversion_enhanced(df: pd.DataFrame, conversion: dict):
    """Enhanced type conversion with better error handling"""
    df = df.copy()
    for col, target_type in conversion.items():
        if col not in df.columns:
            continue
        try:
            if target_type in ('int64', 'int'):
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            elif target_type in ('float64', 'float'):
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            elif target_type in ('datetime', 'datetime64[ns]'):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif target_type in ('boolean', 'bool'):
                truthy = {'true','yes','1','y','t'}
                falsey = {'false','no','0','n','f'}
                def _to_bool(v):
                    if pd.isna(v):
                        return pd.NA
                    if isinstance(v, bool):
                        return v
                    s = str(v).strip().lower()
                    if s in truthy:
                        return True
                    if s in falsey:
                        return False
                    try:
                        f = float(s)
                        if f == 1:
                            return True
                        if f == 0:
                            return False
                    except Exception:
                        pass
                    return pd.NA
                df[col] = df[col].map(_to_bool).astype('boolean')
            elif target_type in ('object', 'text', 'string'):
                df[col] = df[col].astype('object')
            else:
                df[col] = df[col].astype('object')
        except Exception:
            pass
    return df

def profile_nulls_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced null profiling with detailed analysis"""
    total = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        ratio = (nulls / total) * 100 if total > 0 else 0.0
        dtype_str = str(s.dtype)
        nunique = int(s.nunique(dropna=True))
        rows.append({
            'column': col,
            'dtype': dtype_str,
            'null_count': nulls,
            'null_pct': ratio,
            'unique': nunique,
        })
    return pd.DataFrame(rows)

def suggest_null_strategy_enhanced(col_name: str, s: pd.Series) -> str:
    """Enhanced null strategy suggestion with smart logic"""
    name = str(col_name).lower()
    dtype_str = str(s.dtype)
    null_ratio = (s.isna().mean() * 100.0) if len(s) else 0.0

    if dtype_str == 'object':
        s = s.replace(["", " ", "NA", "N/A", "NULL", "-", "--", "NaN", "None"], pd.NA)

    if dtype_str in ('boolean', 'bool') or name.endswith('_flag') or name.startswith('is_') or name.startswith('has_'):
        return 'mode'

    if dtype_str.startswith('datetime') or any(tok in name for tok in ['date', 'time', 'timestamp', 'created', 'updated']):
        return 'ffill'

    if dtype_str in ('int64', 'Int64', 'float64') or any(tok in name for tok in ['count','score','price','amount','quantity','age','id']):
        if null_ratio < 5:
            return 'median'
        if null_ratio < 30:
            return 'median'
        if any(tok in name for tok in ['count','qty','quantity','rate']):
            return 'zero'
        return 'No change'

    if dtype_str == 'object':
        if null_ratio < 10:
            return 'mode'
        return 'unknown'

    return 'No change'

def apply_null_strategies_enhanced(df: pd.DataFrame, strategies: dict, add_flags: bool = True) -> pd.DataFrame:
    """Enhanced null strategy application with better error handling"""
    out = df.copy()
    for col, strat in strategies.items():
        if col not in out.columns:
            continue
        s = out[col]
        if add_flags:
            flag_name = f"{col}__was_null"
            out[flag_name] = s.isna()
        try:
            if strat == 'drop':
                out = out.dropna(subset=[col])
                continue
            if strat == 'mean':
                out[col] = s.fillna(s.astype('float64').mean())
            elif strat == 'median':
                out[col] = s.fillna(s.astype('float64').median())
            elif strat == 'mode':
                mode_val = s.mode(dropna=True)
                mode_val = mode_val.iloc[0] if not mode_val.empty else s.dropna().iloc[0] if s.dropna().size else s
                out[col] = s.fillna(mode_val)
            elif strat == 'zero':
                out[col] = s.fillna(0)
            elif strat == 'unknown':
                out[col] = s.fillna('Unknown')
            elif strat == 'ffill':
                out[col] = s.fillna(method='ffill')
                out[col] = out[col].fillna(method='bfill')
            elif strat == 'bfill':
                out[col] = s.fillna(method='bfill')
                out[col] = out[col].fillna(method='ffill')
        except Exception:
            pass
    return out

def analyze_duplicates_enhanced(df: pd.DataFrame) -> dict:
    """Enhanced duplicate row analysis with detailed statistics"""
    total_rows = len(df)
    duplicate_mask = df.duplicated(keep=False)
    duplicate_rows = df[duplicate_mask]
    unique_duplicate_groups = df.duplicated(keep='first').sum()
    
    # Find duplicate groups
    duplicate_groups = []
    if len(duplicate_rows) > 0:
        # Group by all columns to find duplicate sets
        grouped = df.groupby(list(df.columns))
        for name, group in grouped:
            if len(group) > 1:
                duplicate_groups.append({
                    'values': dict(zip(df.columns, name)),
                    'count': len(group),
                    'indices': group.index.tolist()
                })
    
    return {
        'total_rows': total_rows,
        'duplicate_rows_count': len(duplicate_rows),
        'unique_duplicate_groups': unique_duplicate_groups,
        'duplicate_percentage': (len(duplicate_rows) / total_rows * 100) if total_rows > 0 else 0,
        'duplicate_groups': duplicate_groups[:10],  # Show first 10 groups
        'has_duplicates': len(duplicate_rows) > 0
    }

def remove_duplicates_enhanced(df: pd.DataFrame, strategy: str = 'keep_first') -> pd.DataFrame:
    """Enhanced duplicate removal with multiple strategies"""
    if strategy == 'keep_first':
        return df.drop_duplicates(keep='first')
    elif strategy == 'keep_last':
        return df.drop_duplicates(keep='last')
    elif strategy == 'remove_all':
        # Remove all rows that have duplicates
        return df.drop_duplicates(keep=False)
    elif strategy == 'keep_all':
        # Don't remove any duplicates
        return df
    else:
        # Default to keep_first
        return df.drop_duplicates(keep='first')

def remove_stopwords_from_text_column_enhanced(df, remove_stopwords=True):
    """Enhanced stopwords removal with spaCy integration"""
    if not remove_stopwords:
        return df
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except ImportError:
        logger.warning("spaCy not available for stopwords removal")
        return df
    
    text_cols = [col for col in df.select_dtypes(include=["object"]).columns 
                 if df[col].dropna().astype(str).str.match('.[a-zA-Z]+.').any()]
    if not text_cols:
        return df

    def process_text(text):
        doc = nlp(str(text))
        filtered_tokens = [token.text for token in doc if not token.is_stop]
        return " ".join(filtered_tokens)

    for col in text_cols:
        df[col] = df[col].apply(process_text)
    return df

def process_text_enhanced(df, method):
    """Enhanced text processing with lemmatization and stemming"""
    try:
        import spacy
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        stemmer = PorterStemmer()
    except ImportError:
        logger.warning("spaCy or NLTK not available for text processing")
        return df

    def lemmatize_text(text):
        doc = nlp(str(text))
        return " ".join([token.text if token.lemma_ == '-PRON-' else token.lemma_ for token in doc])

    def stem_text(text):
        words = word_tokenize(str(text))
        return " ".join([stemmer.stem(word) for word in words])

    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        if method == 'lemmatize':
            df[col] = df[col].apply(lemmatize_text)
        elif method == 'stem':
            df[col] = df[col].apply(stem_text)
    return df

def preprocess_csv_enhanced(input_obj, fill_null_strategy=None, type_conversions=None, remove_stopwords_flag=False):
    """Enhanced CSV preprocessing with all advanced features"""
    if isinstance(input_obj, pd.DataFrame):
        df = input_obj.copy()
    else:
        df = pd.read_csv(input_obj)
    
    df.columns = validate_and_normalize_headers_enhanced(df.columns)

    # Normalize text columns
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in text_cols:
        df[col] = normalize_text_column_enhanced(df[col])

    # Apply type conversions
    if type_conversions:
        df = apply_type_conversion_enhanced(df, type_conversions)

    # Remove stopwords if flagged
    if remove_stopwords_flag:
        df = remove_stopwords_from_text_column_enhanced(df, remove_stopwords=True)

    file_meta = {}
    numeric_metadata = []
    return df, file_meta, numeric_metadata

# -----------------------------
# 🔹 Enhanced Chunking Functions
# -----------------------------
def estimate_token_count_enhanced(text: str) -> int:
    """Enhanced token count estimation"""
    if not text:
        return 0
    return max(1, len(text) // 4)

def chunk_fixed_enhanced(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Enhanced fixed chunking with better text splitting"""
    start_time = time.time()
    rows = []
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        rows.append(row_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )

    text = "\n".join(rows)
    chunks = splitter.split_text(text)
    logger.info(f"Enhanced fixed chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks

def chunk_semantic_cluster_enhanced(df: pd.DataFrame, n_clusters: int = 10) -> List[str]:
    """Enhanced semantic clustering with better batch processing"""
    start_time = time.time()
    sentences = df.astype(str).agg(" ".join, axis=1).tolist()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    if len(sentences) > 1000:
        embs = []
        for i in range(0, len(sentences), 1000):
            batch = sentences[i:i + 1000]
            batch_embs = model.encode(batch)
            embs.extend(batch_embs)
        embs = np.array(embs)
    else:
        embs = model.encode(sentences)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=min(n_clusters, len(sentences)), random_state=42)
    labels = kmeans.fit_predict(embs)

    grouped = {}
    for sent, lab in zip(sentences, labels):
        grouped.setdefault(int(lab), []).append(sent)

    chunks = [" ".join(v) for v in grouped.values()]
    logger.info(f"Enhanced semantic clustering completed in {time.time() - start_time:.2f}s, created {len(chunks)} clusters")
    return chunks

def document_based_chunking_enhanced(df: pd.DataFrame, key_column: str, token_limit: int = 2000, preserve_headers: bool = True) -> Tuple[List[str], List[dict]]:
    """Enhanced document-based chunking with better token counting and grouping"""
    start_time = time.time()

    if key_column not in df.columns:
        raise ValueError(f"Key column '{key_column}' not found in dataframe")

    chunks = []
    metas = []
    grouped = df.groupby(key_column)
    chunk_index = 0

    for key_value, group in grouped:
        if preserve_headers:
            headers = " | ".join(group.columns.astype(str))
            rows_text = []
            for _, row in group.iterrows():
                row_text = " | ".join([f"{col}:{val}" for col, val in row.items() if pd.notna(val)])
                rows_text.append(row_text)
            group_text = f"HEADERS: {headers}\n" + "\n".join(rows_text)
        else:
            rows_text = []
            for _, row in group.iterrows():
                row_text = " | ".join([f"{col}:{val}" for col, val in row.items() if pd.notna(val)])
                rows_text.append(row_text)
            group_text = "\n".join(rows_text)

        token_count = estimate_token_count_enhanced(group_text)

        if token_count <= token_limit:
            chunks.append(group_text)
            metas.append({
                'chunk_index': chunk_index,
                'key_column': key_column,
                'key_value': str(key_value),
                'chunking_method': 'document_based',
                'token_count': token_count,
                'token_limit': token_limit,
                'group_size': len(group),
                'is_subchunk': False
            })
            chunk_index += 1
        else:
            avg_tokens_per_row = token_count / len(group)
            rows_per_chunk = max(1, min(len(group), int(token_limit / avg_tokens_per_row)))

            if len(group) / rows_per_chunk > 10:
                rows_per_chunk = max(rows_per_chunk, len(group) // 10)

            for i in range(0, len(group), rows_per_chunk):
                end_idx = min(i + rows_per_chunk, len(group))
                sub_group = group.iloc[i:end_idx]

                if preserve_headers:
                    headers = " | ".join(sub_group.columns.astype(str))
                    rows_text = []
                    for _, row in sub_group.iterrows():
                        row_text = " | ".join([f"{col}:{val}" for col, val in row.items() if pd.notna(val)])
                        rows_text.append(row_text)
                    sub_text = f"HEADERS: {headers}\n" + "\n".join(rows_text)
                else:
                    rows_text = []
                    for _, row in sub_group.iterrows():
                        row_text = " | ".join([f"{col}:{val}" for col, val in row.items() if pd.notna(val)])
                        rows_text.append(row_text)
                    sub_text = "\n".join(rows_text)

                sub_token_count = estimate_token_count_enhanced(sub_text)
                chunks.append(sub_text)
                metas.append({
                    'chunk_index': chunk_index,
                    'key_column': key_column,
                    'key_value': str(key_value),
                    'chunking_method': 'document_based',
                    'token_count': sub_token_count,
                    'token_limit': token_limit,
                    'group_size': len(group),
                    'subchunk_index': (i // rows_per_chunk) + 1,
                    'total_subchunks': (len(group) + rows_per_chunk - 1) // rows_per_chunk,
                    'is_subchunk': True,
                    'rows_in_chunk': len(sub_group)
                })
                chunk_index += 1

    logger.info(f"Enhanced document-based chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks, metas

def chunk_recursive_keyvalue_enhanced(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Enhanced recursive key-value chunking with better handling"""
    start_time = time.time()
    rows = []
    for _, row in df.iterrows():
        kv_pairs = [f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c])]
        rows.append(" | ".join(kv_pairs))

    big_text = "\n".join(rows)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " | ", " ", ""]
    )

    chunks = splitter.split_text(big_text)
    logger.info(f"Enhanced recursive KV chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks

# -----------------------------
# 🔹 Enhanced Embedding Functions
# -----------------------------
class OpenAIEmbeddingAPIEnhanced:
    """Enhanced OpenAI-compatible embedding API"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None, base_url: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.is_local = not api_key
    
    def encode(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Encode texts using OpenAI API or local fallback"""
        if self.is_local:
            from sentence_transformers import SentenceTransformer
            local_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            embeddings = local_model.encode(texts, batch_size=batch_size)
            return np.array(embeddings).astype("float32")
        else:
            import openai
            openai.api_key = self.api_key
            if self.base_url:
                openai.base_url = self.base_url
            
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                try:
                    response = openai.embeddings.create(
                        model=self.model_name,
                        input=batch_texts
                    )
                    batch_embeddings = [data.embedding for data in response.data]
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"OpenAI API error: {e}")
                    from sentence_transformers import SentenceTransformer
                    local_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                    fallback_embeddings = local_model.encode(batch_texts)
                    embeddings.extend(fallback_embeddings)
            
            return np.array(embeddings).astype("float32")

def parallel_embed_texts_enhanced(chunks: List[str], model_name: str = "paraphrase-MiniLM-L6-v2", batch_size: int = EMBEDDING_BATCH_SIZE, num_workers: int = PARALLEL_WORKERS):
    """Enhanced parallel embedding for faster processing"""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    
    def embed_batch(batch_chunks: List[str]):
        return model.encode(batch_chunks, batch_size=batch_size)
    
    per_batch = max(1, len(chunks) // max(1, num_workers))
    chunk_batches = [chunks[i:i + per_batch] for i in range(0, len(chunks), per_batch)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(embed_batch, chunk_batches))
    
    embeddings = np.vstack(results)
    return model, embeddings

def embed_texts_enhanced(
    chunks: List[str],
    model_name: str = "paraphrase-MiniLM-L6-v2",
    openai_api_key: str = None,
    openai_base_url: str = None,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    use_parallel: bool = True
) -> Tuple[object, np.ndarray]:
    """Enhanced embedding with parallel processing and OpenAI support"""
    start_time = time.time()
    
    if use_parallel and len(chunks) > 500 and not openai_api_key and "text-embedding" not in model_name.lower():
        logger.info(f"Using parallel processing for {len(chunks)} chunks")
        model, embeddings = parallel_embed_texts_enhanced(chunks, model_name, batch_size)
        logger.info(f"Enhanced parallel embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
        return model, embeddings
    
    if openai_api_key or "text-embedding" in model_name.lower():
        openai_model = OpenAIEmbeddingAPIEnhanced(
            model_name=model_name if "text-embedding" in model_name.lower() else "text-embedding-ada-002",
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        embeddings = openai_model.encode(chunks, batch_size=batch_size)
        model = openai_model
        logger.info(f"Enhanced OpenAI embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, batch_size=batch_size)
        logger.info(f"Enhanced local embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
    
    return model, np.array(embeddings).astype("float32")

# -----------------------------
# 🔹 Enhanced Storage Functions
# -----------------------------
def store_chroma_enhanced(chunks: List[str], embeddings: np.ndarray, collection_name: str = "chunks_collection", metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Enhanced ChromaDB storage with metadata support"""
    start_time = time.time()
    try:
        import chromadb
    except Exception as e:
        raise ImportError("chromadb is required for Chroma storage. Please install 'chromadb'.") from e

    client = chromadb.PersistentClient(path="chromadb_store")
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name)
    
    batch_size = 1000
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_ids = [str(j) for j in range(i, end_idx)]
        
        # Add metadata if provided
        batch_metadata = None
        if metadata and i < len(metadata):
            batch_metadata = metadata[i:end_idx]
        
        col.add(
            ids=batch_ids,
            documents=batch_chunks,
            embeddings=batch_embeddings.tolist(),
            metadatas=batch_metadata
        )
    
    logger.info(f"Enhanced Chroma storage completed in {time.time() - start_time:.2f}s, stored {len(chunks)} vectors")
    return {"type": "chroma", "collection": col, "collection_name": collection_name}

def create_metadata_index(metadata: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[int]]]:
    """Create fast metadata index for filtering"""
    metadata_index = {}
    for i, meta in enumerate(metadata):
        for key, value in meta.items():
            if key not in metadata_index:
                metadata_index[key] = {}
            # Convert value to string for consistent indexing
            str_value = str(value)
            if str_value not in metadata_index[key]:
                metadata_index[key][str_value] = []
            metadata_index[key][str_value].append(i)
    return metadata_index

def apply_metadata_filter(metadata_index: Dict[str, Dict[str, List[int]]], filter_dict: Dict[str, Any]) -> List[int]:
    """Apply metadata filter and return matching indices"""
    if not filter_dict:
        # No filter - return all indices
        all_indices = set()
        for key_dict in metadata_index.values():
            for indices_list in key_dict.values():
                all_indices.update(indices_list)
        return list(all_indices)
    
    # Start with all indices from the first filter key
    matching_indices = None
    
    # Apply each filter condition
    for key, value in filter_dict.items():
        if key in metadata_index:
            str_value = str(value)
            if str_value in metadata_index[key]:
                current_indices = set(metadata_index[key][str_value])
                if matching_indices is None:
                    matching_indices = current_indices
                else:
                    # Intersect with current matching indices
                    matching_indices = matching_indices.intersection(current_indices)
            else:
                # No matches for this value
                return []
        else:
            # Key not in metadata index
            return []
    
    return list(matching_indices) if matching_indices is not None else []

def query_faiss_with_metadata(index, faiss_data: Dict[str, Any], query_embedding: np.ndarray, k: int = 5, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Query FAISS with metadata filtering support"""
    # Get more results if filtering is applied
    search_k = k * 3 if metadata_filter else k
    
    # Vector search in FAISS
    distances, indices = index.search(query_embedding, search_k)
    
    results = []
    chunks = faiss_data.get("documents", [])
    metadata = faiss_data.get("metadata", [])
    metadata_index = faiss_data.get("metadata_index", {})
    
    # Apply metadata filtering if provided
    if metadata_filter:
        matching_indices = apply_metadata_filter(metadata_index, metadata_filter)
        if not matching_indices:
            return []  # No matches after filtering
        
        # Filter FAISS results by metadata
        filtered_results = []
        for i, idx in enumerate(indices[0]):
            if idx in matching_indices and len(filtered_results) < k:
                similarity = 1 / (1 + distances[0][i])
                filtered_results.append({
                    "rank": len(filtered_results) + 1,
                    "content": chunks[idx] if idx < len(chunks) else "",
                    "similarity": float(similarity),
                    "distance": float(distances[0][i]),
                    "metadata": metadata[idx] if idx < len(metadata) else {}
                })
        return filtered_results
    
    # No filtering - return top k results
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if i >= k:
            break
        if idx < len(chunks):
            similarity = 1 / (1 + distance)
            results.append({
                "rank": i + 1,
                "content": chunks[idx],
                "similarity": float(similarity),
                "distance": float(distance),
                "metadata": metadata[idx] if idx < len(metadata) else {}
            })
    
    return results

def store_faiss_enhanced(chunks: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Enhanced FAISS storage with metadata support and filtering capabilities"""
    start_time = time.time()
    try:
        import faiss
        import pickle
        import os
    except Exception as e:
        raise ImportError("faiss is required for FAISS storage. Please install 'faiss-cpu' or 'faiss-gpu'.") from e

    d = int(embeddings.shape[1]) if len(embeddings.shape) == 2 else 0
    index = faiss.IndexFlatL2(d)
    
    batch_size = 10000
    total = embeddings.shape[0]
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        batch_embeddings = embeddings[i:end_idx]
        index.add(batch_embeddings)
    
    # Enhanced metadata handling
    if metadata is None:
        metadata = [{"chunk_id": str(i)} for i in range(len(chunks))]
    
    # Create metadata index for fast filtering
    metadata_index = create_metadata_index(metadata)
    
    faiss_data = {
        "documents": chunks,
        "metadata": metadata,
        "metadata_index": metadata_index,
        "total_vectors": len(chunks),
        "embedding_dim": d
    }
    
    os.makedirs("faiss_store", exist_ok=True)
    faiss.write_index(index, "faiss_store/index.faiss")
    with open("faiss_store/data.pkl", "wb") as f:
        pickle.dump(faiss_data, f)
    
    logger.info(f"Enhanced FAISS storage completed in {time.time() - start_time:.2f}s, stored {embeddings.shape[0]} vectors with metadata indexing")
    return {"type": "faiss", "index": index, "data": faiss_data, "metadata_index": metadata_index}

# -----------------------------
# 🔹 Main Deep Config Pipeline Function
# -----------------------------
def run_deep_config_pipeline(df, config_dict, file_info=None):
    """
    Comprehensive deep config pipeline that handles:
    1. Enhanced preprocessing
    2. Advanced chunking with metadata
    3. Parallel embedding generation
    4. Enhanced vector storage
    """
    global current_model, current_store_info, current_chunks, current_embeddings, current_df
    
    current_df = df.copy()
    set_file_info(file_info)
    
    start_time = time.time()
    
    # Step 1: Enhanced Preprocessing
    preprocessing_config = config_dict.get('preprocessing', {})
    if preprocessing_config:
        df, file_meta, numeric_meta = preprocess_csv_enhanced(
            df,
            fill_null_strategy=preprocessing_config.get('fill_null_strategy'),
            type_conversions=preprocessing_config.get('type_conversions'),
            remove_stopwords_flag=preprocessing_config.get('remove_stopwords_flag', False)
        )
    
    # Step 2: Enhanced Chunking
    chunking_config = config_dict.get('chunking', {})
    chunk_method = chunking_config.get('method', 'fixed')
    
    if chunk_method == "fixed":
        chunks = chunk_fixed_enhanced(df, 
                                    chunking_config.get('chunk_size', 400), 
                                    chunking_config.get('overlap', 50))
        chunk_metadata = [{"chunk_id": f"fixed_{i:04d}", "method": "fixed"} for i in range(len(chunks))]
    elif chunk_method == "recursive":
        chunks = chunk_recursive_keyvalue_enhanced(df, 
                                                 chunking_config.get('chunk_size', 400), 
                                                 chunking_config.get('overlap', 50))
        chunk_metadata = [{"chunk_id": f"kv_{i:04d}", "method": "recursive_kv"} for i in range(len(chunks))]
    elif chunk_method == "semantic":
        chunks = chunk_semantic_cluster_enhanced(df, chunking_config.get('n_clusters', 10))
        chunk_metadata = [{"chunk_id": f"sem_cluster_{i:04d}", "method": "semantic_cluster"} for i in range(len(chunks))]
    elif chunk_method == "document":
        chunks, chunk_metadata = document_based_chunking_enhanced(
            df, 
            chunking_config.get('key_column', df.columns[0]),
            chunking_config.get('token_limit', 2000),
            chunking_config.get('preserve_headers', True)
        )
    
    # Step 3: Enhanced Embedding
    embedding_config = config_dict.get('embedding', {})
    model, embeddings = embed_texts_enhanced(
        chunks,
        embedding_config.get('model_name', 'paraphrase-MiniLM-L6-v2'),
        embedding_config.get('openai_api_key'),
        embedding_config.get('openai_base_url'),
        embedding_config.get('batch_size', 64),
        embedding_config.get('use_parallel', True)
    )
    
    # Step 4: Enhanced Storage
    storage_config = config_dict.get('storage', {})
    storage_choice = storage_config.get('type', 'chroma')
    collection_name = storage_config.get('collection_name', f"deep_config_{int(time.time())}")
    
    if storage_choice == "faiss":
        store = store_faiss_enhanced(chunks, embeddings, chunk_metadata)
    else:
        store = store_chroma_enhanced(chunks, embeddings, collection_name, chunk_metadata)
    
    # Store for retrieval
    current_model = model
    current_store_info = store
    current_chunks = chunks
    current_embeddings = embeddings
    
    # Save state to disk for persistence across API calls
    save_state()
    
    processing_time = time.time() - start_time
    
    return {
        "rows": len(df), 
        "chunks": len(chunks), 
        "stored": store["type"],
        "embedding_model": embedding_config.get('model_name', 'paraphrase-MiniLM-L6-v2'),
        "retrieval_ready": True,
        "processing_time": processing_time,
        "chunk_metadata": chunk_metadata,
        "preprocessing_applied": bool(preprocessing_config),
        "enhanced_pipeline": True
    }