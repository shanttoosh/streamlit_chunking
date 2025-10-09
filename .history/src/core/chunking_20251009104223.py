# Text Chunking Module
import pandas as pd
import numpy as np
import time
import logging
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.cluster import KMeans
from .preprocessing import estimate_token_count

logger = logging.getLogger(__name__)

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

# Enhanced chunking functions for deep config mode
def chunk_fixed_enhanced(df: pd.DataFrame, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Enhanced fixed chunking with better token counting"""
    start_time = time.time()
    
    # Convert dataframe to text rows
    rows = []
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        rows.append(row_text)
    
    # Use improved text splitter
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
    """Enhanced semantic clustering with better error handling"""
    start_time = time.time()
    sentences = df.astype(str).agg(" ".join, axis=1).tolist()
    
    try:
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
            grouped.setdefault(int(lab), []).append(sent)

        chunks = [" ".join(v) for v in grouped.values()]
        logger.info(f"Enhanced semantic clustering completed in {time.time() - start_time:.2f}s, created {len(chunks)} clusters")
        return chunks
    except Exception as e:
        logger.error(f"Semantic clustering failed: {e}")
        # Fallback to fixed chunking
        return chunk_fixed_enhanced(df, chunk_size=400, overlap=50)

def document_based_chunking_enhanced(df: pd.DataFrame, key_column: str, token_limit: int = 2000, preserve_headers: bool = True) -> Tuple[List[str], List[dict]]:
    """Enhanced document-based chunking with better error handling"""
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
        
        # Enhanced token count estimation
        token_count = estimate_token_count(group_text)
        
        # If group fits within token limit, use as single chunk
        if token_count <= token_limit:
            chunks.append(group_text)
            metas.append({
                'chunk_index': chunk_index,
                'key_column': key_column,
                'key_value': str(key_value),
                'chunking_method': 'document_based_enhanced',
                'token_count': token_count,
                'token_limit': token_limit,
                'group_size': len(group),
                'is_subchunk': False
            })
            chunk_index += 1
        else:
            # Enhanced sub-chunking: calculate optimal rows per chunk
            avg_tokens_per_row = token_count / len(group)
            rows_per_chunk = max(1, min(len(group), int(token_limit / avg_tokens_per_row)))
            
            # Ensure we don't create too many tiny chunks
            if len(group) / rows_per_chunk > 10:
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
                    'chunking_method': 'document_based_enhanced',
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
    """Enhanced recursive key-value chunking"""
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
        separators=["\n\n", "\n", " | ", " ", ""]
    )
    
    chunks = splitter.split_text(big_text)
    logger.info(f"Enhanced recursive KV chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks