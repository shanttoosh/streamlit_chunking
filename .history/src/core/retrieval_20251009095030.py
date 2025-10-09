# Retrieval Functions
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def retrieve_similar(query: str, k: int = 5, current_model=None, current_store_info=None, 
                    current_chunks=None, current_embeddings=None):
    """Retrieve similar chunks using the current stored embeddings"""
    start_time = time.time()
    
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
        # FAISS retrieval
        index = current_store_info["index"]
        faiss_data = current_store_info.get("data", {})
        
        # Use enhanced query function if available
        if "data" in current_store_info:
            results = query_faiss_with_metadata(index, faiss_data, query_arr, k)
        else:
            # Basic FAISS query
            distances, indices = index.search(query_arr, k)
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(current_chunks):
                    similarity = 1 / (1 + distance)
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
