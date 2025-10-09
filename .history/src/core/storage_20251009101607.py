# Vector Storage Module
import numpy as np
import time
import logging
import os
import pickle
from typing import List, Dict, Any, Optional
import chromadb
import faiss

logger = logging.getLogger(__name__)

def store_chroma(chunks, embeddings, collection_name="chunks_collection"):
    """Store embeddings in ChromaDB"""
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

def store_chroma_with_metric(chunks, embeddings, collection_name, metric="cosine"):
    """Store embeddings in ChromaDB with specific metric"""
    start_time = time.time()
    client = chromadb.PersistentClient(path="chromadb_store")
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    
    # Create collection with specific distance function
    distance_function = "cosine" if metric == "cosine" else "l2"
    col = client.create_collection(collection_name, metadata={"hnsw:space": distance_function})
    
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
    return {"type": "chroma", "collection": col, "collection_name": collection_name, "metric": metric}

def store_faiss_with_metric(embeddings, metric="cosine"):
    """Store embeddings in FAISS with specific metric"""
    start_time = time.time()
    d = embeddings.shape[1]
    
    # Choose index based on metric
    if metric == "cosine":
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
    elif metric == "dot":
        index = faiss.IndexFlatIP(d)  # Inner product
    else:  # euclidean
        index = faiss.IndexFlatL2(d)  # L2 distance
    
    # Add in batches for large embeddings
    batch_size = 10000
    for i in range(0, embeddings.shape[0], batch_size):
        end_idx = min(i + batch_size, embeddings.shape[0])
        batch_embeddings = embeddings[i:end_idx]
        index.add(batch_embeddings)
    
    logger.info(f"FAISS storage with {metric} metric completed in {time.time() - start_time:.2f}s, stored {embeddings.shape[0]} vectors")
    return {"type": "faiss", "index": index, "metric": metric}

# Enhanced storage functions for deep config mode
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