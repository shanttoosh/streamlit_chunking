# Processing Pipelines
import pandas as pd
import numpy as np
import time
import logging
import pickle
import os
from typing import Dict, Any, Optional, Tuple, List
from .preprocessing import (
    preprocess_auto_fast, preprocess_optimized_fast, preprocess_auto_config1,
    validate_and_normalize_headers, normalize_text_column, preprocess_csv_enhanced
)
from .chunking import (
    chunk_fixed, chunk_recursive_keyvalue, chunk_semantic_cluster, document_based_chunking,
    chunk_fixed_enhanced, chunk_semantic_cluster_enhanced, document_based_chunking_enhanced,
    chunk_recursive_keyvalue_enhanced
)
from .embedding import embed_texts, embed_texts_enhanced
from .storage import (
    store_faiss, store_chroma, store_faiss_with_metric, store_chroma_with_metric,
    store_faiss_enhanced, store_chroma_enhanced
)
from .database import set_file_info
# from ..config.settings import settings

logger = logging.getLogger(__name__)

# Global variables to store current state for retrieval
current_model = None
current_store_info = None
current_chunks = None
current_embeddings = None
current_metadata = None
current_df = None
current_file_info = None

# Performance Configuration
EMBEDDING_BATCH_SIZE = 256
PARALLEL_WORKERS = 6
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

def set_file_info(file_info: Dict):
    """Store file information"""
    global current_file_info
    current_file_info = file_info

def get_file_info():
    """Get stored file information"""
    global current_file_info
    return current_file_info or {}

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

def run_fast_pipeline(df, db_type="sqlite", db_config=None, file_info=None, 
                     use_openai=False, openai_api_key=None, openai_base_url=None,
                     use_turbo=False, batch_size=EMBEDDING_BATCH_SIZE):
    """Fast mode pipeline"""
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
    """Config-1 mode pipeline"""
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
        store = store_faiss_with_metric(embs, retrieval_metric)
    else:
        store = store_chroma_with_metric(chunks, embs, f"config1_{int(time.time())}", retrieval_metric)

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

def run_deep_config_pipeline(df, config_dict, file_info=None):
    """Deep config mode pipeline"""
    global current_model, current_store_info, current_chunks, current_embeddings, current_df
    
    current_df = df.copy()
    set_file_info(file_info)
    
    start_time = time.time()
    
    # Step 1: Enhanced Preprocessing
    preprocessing_config = config_dict.get('preprocessing', {})
    if preprocessing_config:
        # Apply enhanced preprocessing
        df = preprocess_csv_enhanced(
            df,
            fill_null_strategy=preprocessing_config.get('fill_null_strategy'),
            type_conversions=preprocessing_config.get('type_conversions'),
            remove_stopwords_flag=preprocessing_config.get('remove_stopwords', False)
        )
    
    # Step 2: Enhanced Chunking
    chunking_config = config_dict.get('chunking', {})
    chunk_method = chunking_config.get('method', 'fixed')
    
    if chunk_method == "fixed":
        chunks = chunk_fixed_enhanced(df, 
                            chunking_config.get('chunk_size', 400), 
                            chunking_config.get('overlap', 50))
    elif chunk_method == "recursive":
        chunks = chunk_recursive_keyvalue_enhanced(df, 
                                         chunking_config.get('chunk_size', 400), 
                                         chunking_config.get('overlap', 50))
    elif chunk_method == "semantic":
        chunks = chunk_semantic_cluster_enhanced(df, chunking_config.get('n_clusters', 10))
    elif chunk_method == "document":
        chunks, metadata = document_based_chunking_enhanced(
            df, 
            chunking_config.get('key_column', df.columns[0]),
            chunking_config.get('token_limit', 2000),
            chunking_config.get('preserve_headers', True)
        )
    else:
        chunks = chunk_fixed_enhanced(df, 400, 50)
        metadata = None
    
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
        store = store_faiss_enhanced(chunks, embeddings, metadata)
    else:
        store = store_chroma_enhanced(chunks, embeddings, collection_name, metadata)
    
    # Store for retrieval
    current_model = model
    current_store_info = store
    current_chunks = chunks
    current_embeddings = embeddings
    current_metadata = metadata
    
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
        "enhanced_pipeline": True
    }

# Large file processing functions
def process_large_file(file_path: str, processing_mode: str = "fast", **kwargs):
    """Process large files by handling them in batches"""
    file_size = os.path.getsize(file_path)
    
    from .database import can_load_file
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
    
    from .database import process_large_file_in_batches
    return process_large_file_in_batches(file_path, process_batch)

def process_large_file_config1(file_path: str, **kwargs):
    """Process large file in config1 mode with batches"""
    def process_batch(batch_df):
        # Use config1 preprocessing
        processed_df = preprocess_auto_config1(batch_df)
        chunks = chunk_fixed(processed_df, kwargs.get('chunk_size', 400), kwargs.get('overlap', 50))
        return chunks
    
    from .database import process_large_file_in_batches
    return process_large_file_in_batches(file_path, process_batch)

def process_large_file_deep(file_path: str, **kwargs):
    """Process large file in deep config mode with batches"""
    def process_batch(batch_df):
        # Use deep config preprocessing
        config_dict = kwargs.get('config_dict', {})
        return run_deep_config_pipeline(batch_df, config_dict)
    
    from .database import process_large_file_in_batches
    return process_large_file_in_batches(file_path, process_batch)

def process_file_direct(file_path: str, processing_mode: str = "fast", **kwargs):
    """Process file directly"""
    try:
        df = pd.read_csv(file_path)
        
        if processing_mode == "fast":
            return run_fast_pipeline(df, **kwargs)
        elif processing_mode == "config1":
            return run_config1_pipeline(df, **kwargs)
        elif processing_mode == "deep":
            config_dict = kwargs.get('config_dict', {})
            return run_deep_config_pipeline(df, config_dict)
        else:
            raise ValueError(f"Unknown processing mode: {processing_mode}")
    except Exception as e:
        logger.error(f"Direct file processing failed: {e}")
        return {"error": str(e)}