# Storage Functions
import numpy as np
import time
import logging
import os
import pickle
from typing import List, Dict, Any
import chromadb
import faiss
from ..config.settings import settings

logger = logging.getLogger(__name__)

def store_chroma(chunks, embeddings, collection_name="chunks_collection"):
    """Store embeddings in ChromaDB"""
    start_time = time.time()
    client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
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
    """Store embeddings in FAISS"""
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

def store_faiss_with_metric(embeddings, metric="cosine"):
    """Store embeddings in FAISS with specific metric"""
    start_time = time.time()
    d = embeddings.shape[1]
    
    # Metric handling for FAISS
    if metric == "cosine":
        # Normalize rows to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeddings = embeddings / norms
        index = faiss.IndexFlatIP(d)
        index.add(normalized_embeddings)
    elif metric == "dot":
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
    else:  # euclidean
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
    
    logger.info(f"FAISS storage with {metric} metric completed in {time.time() - start_time:.2f}s, stored {embeddings.shape[0]} vectors")
    return {"type": "faiss", "index": index, "metric": metric}

def store_chroma_with_metric(chunks, embeddings, collection_name="chunks_collection", metric="cosine"):
    """Store embeddings in ChromaDB with specific metric"""
    start_time = time.time()
    
    # Chroma metric space: "cosine", "l2", or "ip"
    space = metric.lower()
    if space == "euclidean":
        space = "l2"
    
    client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    
    col = client.create_collection(collection_name, metadata={"hnsw:space": space})
    
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
    
    logger.info(f"Chroma storage with {metric} metric completed in {time.time() - start_time:.2f}s, stored {len(chunks)} vectors")
    return {"type": "chroma", "collection": col, "collection_name": collection_name, "space": space}
