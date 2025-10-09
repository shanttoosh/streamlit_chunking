# Semantic Search and Retrieval Module
import numpy as np
import time
import logging
import json
import psutil
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from .storage import query_faiss_with_metadata

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

def retrieve_similar(query: str, k: int = 5):
    """Retrieve similar chunks using the current stored embeddings"""
    global current_model, current_store_info, current_chunks, current_embeddings
    
    start_time = time.time()
    
    # Load state from disk if globals are empty
    if current_model is None or current_store_info is None:
        logger.info("Globals empty, attempting to load state from disk")
        from .pipelines import load_state
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

def retrieve_with_metadata(query: str, k: int = 5, metadata_filter: Dict[str, Any] = None):
    """Retrieve similar chunks with metadata filtering"""
    global current_model, current_store_info, current_chunks, current_embeddings
    
    start_time = time.time()
    
    # Load state from disk if globals are empty
    if current_model is None or current_store_info is None:
        logger.info("Globals empty, attempting to load state from disk")
        from .pipelines import load_state
        load_state()
    
    if current_model is None or current_store_info is None:
        return {"error": "No model or store available. Run a pipeline first."}
    
    # Encode query
    if hasattr(current_model, 'encode'):
        query_embedding = current_model.encode([query])
    else:
        query_embedding = current_model.encode([query])
    
    query_arr = np.array(query_embedding).astype("float32")
    
    results = []
    
    if current_store_info["type"] == "faiss":
        # Enhanced FAISS retrieval with metadata filtering
        index = current_store_info["index"]
        faiss_data = current_store_info.get("data", {})
        
        # Use enhanced query function with metadata filter
        results = query_faiss_with_metadata(index, faiss_data, query_arr, k, metadata_filter)
    
    elif current_store_info["type"] == "chroma":
        # Chroma retrieval with metadata filtering
        collection = current_store_info["collection"]
        
        # Build where clause for metadata filtering
        where_clause = None
        if metadata_filter:
            where_clause = metadata_filter
        
        chroma_results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=["documents", "distances", "metadatas"],
            where=where_clause
        )
        
        for i, (doc, distance, metadata) in enumerate(zip(
            chroma_results["documents"][0], 
            chroma_results["distances"][0],
            chroma_results.get("metadatas", [[]])[0]
        )):
            similarity = 1 / (1 + distance)
            results.append({
                "rank": i + 1,
                "content": doc,
                "similarity": float(similarity),
                "distance": float(distance),
                "metadata": metadata or {}
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
                    "distance": float(1 - similarities[idx]),
                    "metadata": {}
                })
    
    logger.info(f"Metadata retrieval completed in {time.time() - start_time:.2f}s, found {len(results)} results")
    return {"query": query, "k": k, "results": results, "metadata_filter": metadata_filter}

def export_chunks():
    """Export current chunks as CSV format"""
    global current_chunks
    if current_chunks:
        import pandas as pd
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

def export_embeddings_text():
    """Export current embeddings as text format"""
    global current_embeddings, current_chunks
    if current_embeddings is not None and current_chunks is not None:
        output_lines = []
        output_lines.append(f"Total Chunks: {len(current_chunks)}")
        output_lines.append(f"Vector Dimension: {current_embeddings.shape[1] if len(current_embeddings.shape) > 1 else 0}")
        output_lines.append("=" * 50)
        
        for i, (chunk, embedding) in enumerate(zip(current_chunks, current_embeddings)):
            output_lines.append(f"Chunk {i + 1}:")
            output_lines.append(f"Text: {chunk[:100]}..." if len(chunk) > 100 else f"Text: {chunk}")
            output_lines.append(f"Embedding: {embedding[:10]}..." if len(embedding) > 10 else f"Embedding: {embedding}")
            output_lines.append("-" * 30)
        
        return "\n".join(output_lines)
    return "No embeddings available"

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

def set_file_info(file_info: Dict):
    """Store file information"""
    global current_file_info
    current_file_info = file_info

def get_file_info():
    """Get stored file information"""
    global current_file_info
    return current_file_info or {}

def get_capabilities():
    """Get system capabilities"""
    return {
        "processing_modes": ["fast", "config1", "deep_config"],
        "chunking_methods": ["fixed", "recursive", "semantic", "document"],
        "embedding_models": ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L6-v2", "text-embedding-ada-002"],
        "storage_options": ["faiss", "chroma"],
        "retrieval_metrics": ["cosine", "dot", "euclidean"],
        "database_support": ["mysql", "postgresql", "sqlite"],
        "large_file_support": True,
        "parallel_processing": True,
        "metadata_filtering": True,
        "openai_compatibility": True
    }