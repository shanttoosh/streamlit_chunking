#
# backend.py
import pandas as pd
import numpy as np
import re
import time
import os
import psutil
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging
import spacy
import chardet
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# Global variables to store current state for retrieval
current_model = None
current_store_info = None
current_chunks = None
current_embeddings = None
current_metadata = None
current_df = None
current_file_info = None

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
# 🔹 Preprocessing
# -----------------------------
def preprocess_basic(df: pd.DataFrame, null_handling="keep", fill_value=None):
    start_time = time.time()
    if null_handling == "drop":
        df = df.dropna().reset_index(drop=True)
    elif null_handling == "fill" and fill_value is not None:
        df = df.fillna(fill_value)
    logger.info(f"Basic preprocessing completed in {time.time() - start_time:.2f}s")
    return df

def preprocess_advanced(df: pd.DataFrame,
                        null_handling="keep", fill_value=None,
                        remove_stopwords=False, lowercase=False,
                        stemming=False, lemmatization=False):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    start_time = time.time()
    stop_words = set(stopwords.words("english")) if remove_stopwords else set()
    ps = PorterStemmer() if stemming else None
    lm = WordNetLemmatizer() if lemmatization else None

    df = preprocess_basic(df, null_handling, fill_value)

    for col in df.select_dtypes(include=["object"]).columns:
        series = df[col].astype(str)

        if lowercase:
            series = series.str.lower()

        if remove_stopwords or stemming or lemmatization:
            new_vals = []
            for text in series:
                tokens = re.findall(r"\w+", text)
                if remove_stopwords:
                    tokens = [t for t in tokens if t not in stop_words]
                if stemming:
                    tokens = [ps.stem(t) for t in tokens]
                if lemmatization:
                    tokens = [lm.lemmatize(t) for t in tokens]
                new_vals.append(" ".join(tokens))
            series = pd.Series(new_vals)

        df[col] = series

    logger.info(f"Advanced preprocessing completed in {time.time() - start_time:.2f}s")
    return df

# -----------------------------
# 🔹 NEW ENHANCED PREPROCESSING SYSTEM (3-TIER)
# -----------------------------

# Initialize spaCy model (load once)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Please install: python -m spacy download en_core_web_sm")
    nlp = None

# Initialize Porter Stemmer
stemmer = PorterStemmer()

def detect_encoding(file_path):
    """Detect file encoding using chardet"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')
    except:
        return 'utf-8'

def remove_html_tags(text):
    """Remove HTML tags from text"""
    if not isinstance(text, str):
        return text
    try:
        return BeautifulSoup(text, "lxml").get_text(separator=' ')
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

def normalize_text_columns(df: pd.DataFrame):
    """Apply text normalization to object columns"""
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        df[col] = df[col].fillna('')
        df[col] = df[col].map(remove_html_tags)
        df[col] = df[col].map(lambda x: x.lower() if isinstance(x, str) else x)
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        df[col] = df[col].map(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else x)
    return df

def create_metadata(df: pd.DataFrame, data_source_info: dict) -> dict:
    """Create comprehensive metadata for processed data"""
    metadata = {
        "data_source": data_source_info,
        "processing_stats": {
            "preprocessing_time": time.time(),
            "rows_processed": df.shape[0],
            "columns_processed": df.shape[1],
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
        },
        "column_info": {}
    }
    
    for col in df.columns:
        metadata["column_info"][col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": float((df[col].isnull().sum() / len(df)) * 100),
            "unique_values": int(df[col].nunique())
        }
    
    return metadata

def preprocess_fast_mode(df: pd.DataFrame, data_source_info: dict):
    """Fast Mode: Automatic default preprocessing"""
    logger.info("Starting Fast Mode preprocessing...")
    start_time = time.time()
    
    original_shape = df.shape
    
    # Default preprocessing steps
    df = validate_and_normalize_headers(df)
    df = normalize_text_columns(df)
    
    # Auto drop rows with all null values
    df = df.dropna(how='all')
    
    elapsed_time = time.time() - start_time
    
    metadata = create_metadata(df, data_source_info)
    metadata["processing_stats"]["preprocessing_mode"] = "fast"
    metadata["processing_stats"]["preprocessing_time"] = elapsed_time
    metadata["processing_stats"]["rows_dropped"] = original_shape[0] - df.shape[0]
    
    logger.info(f"Fast Mode preprocessing completed in {elapsed_time:.2f}s")
    return df, metadata

def create_null_preview_table(df: pd.DataFrame) -> dict:
    """Create null analysis preview table for Config-1 Mode"""
    preview_data = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / len(df)) * 100
        preview_data.append({
            "column_name": col,
            "null_count": int(null_count),
            "null_percentage": round(null_percentage, 2)
        })
    
    # Sort by null percentage descending
    preview_data.sort(key=lambda x: x["null_percentage"], reverse=True)
    
    return {
        "columns": ["column_name", "null_count", "null_percentage"],
        "data": preview_data[:10],  # Top 10 columns with most nulls
        "total_rows": len(df)
    }

def handle_nulls_by_column(df: pd.DataFrame, null_handling_config: dict):
    """Apply null handling configuration per column"""
    df = df.copy()
    
    for col, config in null_handling_config.items():
        if config["method"] == "skip":
            continue
        elif config["method"] == "drop":
            df[col] = df[col].dropna()
        elif config["method"] == "fill":
            if config["fill_method"] == "median" and pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
            elif config["fill_method"] == "mode":
                fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else ""
            elif config["fill_method"] == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
            elif config["fill_method"] == "custom":
                fill_value = config["custom_value"]
            else:
                fill_value = ""
            
            df[col] = df[col].fillna(fill_value)
    
    return df

def preprocess_config1_mode(df: pd.DataFrame, data_source_info: dict, null_handling_config: dict = None):
    """Config-1 Mode: Default preprocessing + null handling with preview"""
    logger.info("Starting Config-1 Mode preprocessing...")
    start_time = time.time()
    
    # Default preprocessing first
    df = validate_and_normalize_headers(df)
    df = normalize_text_columns(df)
    
    # Create null preview for analysis
    null_preview = create_null_preview_table(df)
    
    # Apply null handling if configuration provided
    if null_handling_config:
        original_nulls = df.isnull().sum().sum()
        df = handle_nulls_by_column(df, null_handling_config)
        handled_nulls = original_nulls - df.isnull().sum().sum()
        logger.info(f"Handled {handled_nulls} null values")
    
    elapsed_time = time.time() - start_time
    
    metadata = create_metadata(df, data_source_info)
    metadata["processing_stats"]["preprocessing_mode"] = "config1"
    metadata["processing_stats"]["preprocessing_time"] = elapsed_time
    metadata["null_preview"] = null_preview
    
    logger.info(f"Config-1 Mode preprocessing completed in {elapsed_time:.2f}s")
    return df, metadata

def create_dtype_preview_table(df: pd.DataFrame) -> dict:
    """Create data type analysis preview table for Deep Mode"""
    preview_data = []
    for col in df.columns:
        preview_data.append({
            "column_name": col,
            "current_dtype": str(df[col].dtype),
            "sample_values": df[col].dropna().head(3).tolist()
        })
    
    return {
        "columns": ["column_name", "current_dtype", "sample_values"],
        "data": preview_data[:10],  # Top 10 columns
        "total_rows": len(df)
    }

def apply_data_type_conversions(df: pd.DataFrame, dtype_config: dict):
    """Apply data type conversions per column configuration"""
    df = df.copy()
    
    for col, config in dtype_config.items():
        if config["target_dtype"] == "skip":
            continue
        
        try:
            if config["target_dtype"] == "numeric":
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif config["target_dtype"] == "datetime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif config["target_dtype"] == "bool":
                df[col] = df[col].astype(bool)
            elif config["target_dtype"] == "object":
                df[col] = df[col].astype(str)
        except Exception as e:
            logger.warning(f"Failed to convert column {col} to {config['target_dtype']}: {e}")
    
    return df

def remove_duplicates_config(df: pd.DataFrame, duplicate_config: dict):
    """Handle duplicate removal based on configuration"""
    if duplicate_config["method"] == "remove":
        return df.drop_duplicates(subset=duplicate_config.get("columns", None))
    return df

def apply_text_processing(df: pd.DataFrame, text_config: dict):
    """Apply text processing (stopwords, normalization, stemming, lemmatization)"""
    if not text_config.get("enabled", False):
        return df
    
    df = df.copy()
    text_cols = df.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        if nlp and text_config.get("lemmatization", False):
            df[col] = df[col].apply(lambda x: " ".join([token.lemma_ for token in nlp(str(x))]))
        elif text_config.get("stemming", False):
            df[col] = df[col].apply(lambda x: " ".join([stemmer.stem(word) for word in word_tokenize(str(x))]))
    
    return df

def preprocess_deep_mode(df: pd.DataFrame, data_source_info: dict, 
                        dtype_config: dict = None, null_handling_config: dict = None,
                        duplicate_config: dict = None, text_config: dict = None):
    """Deep Mode: Full customization preprocessing with previews"""
    logger.info("Starting Deep Mode preprocessing...")
    start_time = time.time()
    
    # Default preprocessing first
    df = validate_and_normalize_headers(df)
    df = normalize_text_columns(df)
    
    # Create previews
    dtype_preview = create_dtype_preview_table(df)
    null_preview = create_null_preview_table(df)
    
    # Apply configurations
    if dtype_config:
        df = apply_data_type_conversions(df, dtype_config)
    
    if null_handling_config:
        df = handle_nulls_by_column(df, null_handling_config)
    
    if duplicate_config:
        df = remove_duplicates_config(df, duplicate_config)
    
    if text_config:
        df = apply_text_processing(df, text_config)
    
    elapsed_time = time.time() - start_time
    
    metadata = create_metadata(df, data_source_info)
    metadata["processing_stats"]["preprocessing_mode"] = "deep"
    metadata["processing_stats"]["preprocessing_time"] = elapsed_time
    metadata["dtype_preview"] = dtype_preview
    metadata["null_preview"] = null_preview
    
    logger.info(f"Deep Mode preprocessing completed in {elapsed_time:.2f}s")
    return df, metadata

# -----------------------------
# 🔹 Chunking
# -----------------------------
def chunk_fixed(df: pd.DataFrame, chunk_size=400, overlap=50):
    start_time = time.time()
    text = "\n".join(df.astype(str).agg(" | ".join, axis=1).tolist())
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    logger.info(f"Fixed chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks

def chunk_recursive_keyvalue(df: pd.DataFrame, chunk_size=400, overlap=50):
    start_time = time.time()
    rows = []
    for _, row in df.iterrows():
        kv_pairs = [f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c])]
        rows.append(" | ".join(kv_pairs))
    big_text = "\n".join(rows)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(big_text)
    logger.info(f"Recursive KV chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks

def chunk_semantic_cluster(df: pd.DataFrame, n_clusters=5):
    """Group rows into clusters based on semantic embeddings of rows."""
    start_time = time.time()
    sentences = df.astype(str).agg(" ".join, axis=1).tolist()
    model = SentenceTransformer("all-MiniLM-L6-v2")
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
# 🔹 UPDATED DOCUMENT-BASED CHUNKING (No overlap, grouping by column)
# -----------------------------
def document_based_chunking(df: pd.DataFrame, key_column: str, 
                          token_limit: int = 2000, 
                          preserve_headers: bool = True) -> Tuple[List[str], List[dict]]:
    """
    NEW: Document-based chunking - groups by key column without overlap
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
        
        # Estimate token count (rough approximation: 4 chars per token)
        token_count = len(group_text) // 4
        
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
            # Split group into sub-chunks based on row count (no overlap between chunks)
            # Calculate how many rows per chunk to stay under token limit
            avg_tokens_per_row = token_count / len(group)
            rows_per_chunk = max(1, int(token_limit / avg_tokens_per_row))
            
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
                
                sub_token_count = len(sub_text) // 4
                
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
                    'is_subchunk': True
                })
                chunk_index += 1
    
    logger.info(f"Document-based chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks, metas

# -----------------------------
# 🔹 Embedding + Storage
# -----------------------------
def embed_texts(chunks, model_name="all-MiniLM-L6-v2"):
    start_time = time.time()
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    logger.info(f"Embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
    return model, np.array(embeddings).astype("float32")

def store_chroma(chunks, embeddings, collection_name="chunks_collection"):
    start_time = time.time()
    client = chromadb.PersistentClient(path="chromadb_store")
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name)
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )
    logger.info(f"Chroma storage completed in {time.time() - start_time:.2f}s, stored {len(chunks)} vectors")
    return {"type": "chroma", "collection": col, "collection_name": collection_name}

def store_faiss(embeddings):
    start_time = time.time()
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    logger.info(f"FAISS storage completed in {time.time() - start_time:.2f}s, stored {embeddings.shape[0]} vectors")
    return {"type": "faiss", "index": index}
# 🔹 Retrieval Functions
# -----------------------------
def retrieve_similar(query: str, k: int = 5):
    """Retrieve similar chunks using the current stored embeddings"""
    global current_model, current_store_info, current_chunks, current_embeddings
    
    start_time = time.time()
    
    if current_model is None or current_store_info is None:
        return {"error": "No model or store available. Run a pipeline first."}
    
    # Encode query
    query_embedding = current_model.encode([query])
    query_arr = np.array(query_embedding).astype("float32")
    
    results = []
    
    if current_store_info["type"] == "faiss":
        # FAISS retrieval
        index = current_store_info["index"]
        distances, indices = index.search(query_arr, k)
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(current_chunks):
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                results.append({
                    "rank": i + 1,
                    "content": current_chunks[idx],
                    "similarity": float(similarity),
                    "distance": float(distance)
                })
    
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

# Note: SQL querying support has been removed from the pipeline.

# -----------------------------
# 🔹 Export Functions
# -----------------------------
def export_chunks():
    """Export current chunks as text"""
    global current_chunks
    if current_chunks:
        return "\n\n---\n\n".join(current_chunks)
    return ""

def export_embeddings():
    """Export current embeddings as numpy array"""
    global current_embeddings
    return current_embeddings

# -----------------------------
# 🔹 System Info
# -----------------------------
def get_system_info():
    """Get system information including memory usage"""
    memory = psutil.virtual_memory()
    return {
        "memory_usage": f"{memory.percent}%",
        "available_memory": f"{memory.available / (1024**3):.2f} GB",
        "total_memory": f"{memory.total / (1024**3):.2f} GB"
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
# 🔹 Pipelines
# -----------------------------
def run_fast_pipeline(df, db_type="sqlite", db_config=None, file_info=None):
    global current_model, current_store_info, current_chunks, current_embeddings, current_df
    
    current_df = df.copy()
    
    # Enhanced preprocessing for Fast Mode
    data_source_info = file_info or {}
    df1, metadata = preprocess_fast_mode(df, data_source_info)
    set_file_info(file_info)
    
    # Default: semantic clustering
    chunks = chunk_semantic_cluster(df1)
    model, embs = embed_texts(chunks)
    store = store_faiss(embs)
    
    # Store for retrieval
    current_model = model
    current_store_info = store
    current_chunks = chunks
    current_embeddings = embs
    
    return {
        "rows": len(df1), 
        "chunks": len(chunks), 
        "stored": store["type"],
        "retrieval_ready": True,
        "metadata": metadata
    }

def run_config1_pipeline(df, null_handling, fill_value, chunk_method,
                         chunk_size, overlap, model_choice, storage_choice, db_config=None, file_info=None):
    global current_model, current_store_info, current_chunks, current_embeddings, current_df
    
    current_df = df.copy()
    set_file_info(file_info)
    df1 = preprocess_basic(df, null_handling, fill_value)

    if chunk_method == "fixed":
        chunks = chunk_fixed(df1, chunk_size, overlap)
    elif chunk_method == "recursive":
        chunks = chunk_recursive_keyvalue(df1, chunk_size, overlap)
    elif chunk_method == "semantic":
        chunks = chunk_semantic_cluster(df1)
    else:  # document
        key_column = df1.columns[0] if len(df1.columns) > 0 else "id"
        chunks, _ = document_based_chunking(df1, key_column, token_limit=2000)

    model, embs = embed_texts(chunks, model_choice)
    
    if storage_choice == "faiss":
        store = store_faiss(embs)
    else:
        store = store_chroma(chunks, embs, f"config1_{int(time.time())}")

    # Store for retrieval
    current_model = model
    current_store_info = store
    current_chunks = chunks
    current_embeddings = embs
    
    return {
        "rows": len(df1), 
        "chunks": len(chunks), 
        "stored": store["type"],
        "retrieval_ready": True
    }

def run_deep_pipeline(df, null_handling, fill_value, remove_stopwords,
                      lowercase, stemming, lemmatization,
                      chunk_method, chunk_size, overlap, model_choice, storage_choice, db_config=None, file_info=None):
    global current_model, current_store_info, current_chunks, current_embeddings, current_df
    
    current_df = df.copy()
    set_file_info(file_info)
    df1 = preprocess_advanced(df, null_handling, fill_value,
                              remove_stopwords, lowercase, stemming, lemmatization)

    if chunk_method == "fixed":
        chunks = chunk_fixed(df1, chunk_size, overlap)
    elif chunk_method == "recursive":
        chunks = chunk_recursive_keyvalue(df1, chunk_size, overlap)
    elif chunk_method == "semantic":
        chunks = chunk_semantic_cluster(df1)
    else:  # document
        key_column = df1.columns[0] if len(df1.columns) > 0 else "id"
        chunks, _ = document_based_chunking(df1, key_column, token_limit=2000)

    model, embs = embed_texts(chunks, model_choice)
    
    if storage_choice == "faiss":
        store = store_faiss(embs)
    else:
        store = store_chroma(chunks, embs, f"deep_{int(time.time())}")

    # Store for retrieval
    current_model = model
    current_store_info = store
    current_chunks = chunks
    current_embeddings = embs
    
    return {
        "rows": len(df1), 
        "chunks": len(chunks), 
        "stored": store["type"],
        "retrieval_ready": True
    }        