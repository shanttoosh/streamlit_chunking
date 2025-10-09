# Embedding Generation Module
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from ..config.settings import settings

logger = logging.getLogger(__name__)

# Performance Configuration
EMBEDDING_BATCH_SIZE = 256
PARALLEL_WORKERS = 6

class OpenAIEmbeddingAPI:
    """OpenAI API wrapper for embeddings with local fallback"""
    
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
    """Generate embeddings for text chunks"""
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

# Enhanced embedding functions for deep config mode
class OpenAIEmbeddingAPIEnhanced:
    """Enhanced OpenAI API wrapper for embeddings with local fallback"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None, base_url: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.is_local = not api_key
    
    def encode(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Enhanced encode with better error handling"""
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
                    # Enhanced fallback
                    from sentence_transformers import SentenceTransformer
                    local_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                    fallback_embeddings = local_model.encode(batch_texts)
                    embeddings.extend(fallback_embeddings)
            
            return np.array(embeddings).astype("float32")

def parallel_embed_texts_enhanced(chunks: List[str], model_name: str = "paraphrase-MiniLM-L6-v2", batch_size: int = EMBEDDING_BATCH_SIZE, num_workers: int = PARALLEL_WORKERS):
    """Enhanced parallel embedding with better error handling"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        
        def embed_batch(batch_chunks: List[str]):
            return model.encode(batch_chunks, batch_size=batch_size)
        
        # Split chunks into batches for parallel processing
        chunk_batches = [chunks[i:i + len(chunks)//num_workers] for i in range(0, len(chunks), len(chunks)//num_workers)]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(embed_batch, chunk_batches))
        
        # Combine results
        embeddings = np.vstack(results)
        return model, embeddings
    except Exception as e:
        logger.error(f"Parallel embedding failed: {e}")
        # Fallback to sequential processing
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, batch_size=batch_size)
        return model, embeddings

def embed_texts_enhanced(
    chunks: List[str], 
    model_name: str = "paraphrase-MiniLM-L6-v2", 
    openai_api_key: str = None, 
    openai_base_url: str = None, 
    batch_size: int = EMBEDDING_BATCH_SIZE, 
    use_parallel: bool = True
):
    """Enhanced embedding generation with better error handling"""
    start_time = time.time()
    
    try:
        # Use parallel processing for large files when enabled
        if use_parallel and len(chunks) > 500 and not openai_api_key and "text-embedding" not in model_name.lower():
            logger.info(f"Using enhanced parallel processing for {len(chunks)} chunks")
            model, embeddings = parallel_embed_texts_enhanced(chunks, model_name, batch_size)
            logger.info(f"Enhanced parallel embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
            return model, embeddings
        
        if openai_api_key or "text-embedding" in model_name.lower():
            # Use OpenAI API
            openai_model = OpenAIEmbeddingAPIEnhanced(
                model_name=model_name if "text-embedding" in model_name.lower() else "text-embedding-ada-002",
                api_key=openai_api_key,
                base_url=openai_base_url
            )
            embeddings = openai_model.encode(chunks, batch_size=batch_size)
            model = openai_model
            logger.info(f"Enhanced OpenAI embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
        else:
            # Use local model
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            embeddings = model.encode(chunks, batch_size=batch_size)
            logger.info(f"Enhanced local embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
        
        return model, np.array(embeddings).astype("float32")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        # Ultimate fallback
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        embeddings = model.encode(chunks, batch_size=batch_size)
        return model, np.array(embeddings).astype("float32")