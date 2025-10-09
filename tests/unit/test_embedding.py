# Unit Tests for Embedding
import pytest
import numpy as np
from src.core.embedding import (
    OpenAIEmbeddingAPI,
    parallel_embed_texts,
    embed_texts
)

class TestEmbedding:
    """Test embedding functions"""
    
    def test_openai_embedding_api_local_fallback(self):
        """Test OpenAI API with local fallback"""
        api = OpenAIEmbeddingAPI(model_name="text-embedding-ada-002", api_key=None)
        
        # Test with local fallback
        texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = api.encode(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384  # paraphrase-MiniLM-L6-v2 dimension
    
    def test_parallel_embed_texts(self, mock_chunks):
        """Test parallel embedding"""
        model, embeddings = parallel_embed_texts(mock_chunks, batch_size=2, num_workers=2)
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(mock_chunks)
        assert embeddings.shape[1] == 384  # paraphrase-MiniLM-L6-v2 dimension
    
    def test_embed_texts_local_model(self, mock_chunks):
        """Test embedding with local model"""
        model, embeddings = embed_texts(
            mock_chunks,
            model_name="paraphrase-MiniLM-L6-v2",
            use_parallel=False
        )
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(mock_chunks)
        assert embeddings.shape[1] == 384
    
    def test_embed_texts_parallel(self, mock_chunks):
        """Test embedding with parallel processing"""
        model, embeddings = embed_texts(
            mock_chunks,
            model_name="paraphrase-MiniLM-L6-v2",
            use_parallel=True
        )
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(mock_chunks)
        assert embeddings.shape[1] == 384
    
    def test_embed_texts_openai(self, mock_chunks):
        """Test embedding with OpenAI API"""
        # This test will use local fallback since no API key is provided
        model, embeddings = embed_texts(
            mock_chunks,
            model_name="text-embedding-ada-002",
            openai_api_key=None,
            use_parallel=False
        )
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(mock_chunks)
    
    def test_embed_texts_empty_list(self):
        """Test embedding with empty text list"""
        model, embeddings = embed_texts([])
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 0
    
    def test_embed_texts_single_text(self):
        """Test embedding with single text"""
        texts = ["Single test sentence."]
        model, embeddings = embed_texts(texts)
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 384
    
    def test_embed_texts_large_batch(self):
        """Test embedding with large batch"""
        texts = [f"Test sentence number {i}." for i in range(100)]
        model, embeddings = embed_texts(texts, batch_size=10)
        
        assert embeddings is not None
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 100
        assert embeddings.shape[1] == 384
    
    def test_openai_embedding_api_batch_processing(self):
        """Test OpenAI API batch processing"""
        api = OpenAIEmbeddingAPI(model_name="text-embedding-ada-002", api_key=None)
        
        texts = [f"Test sentence {i}." for i in range(10)]
        embeddings = api.encode(texts, batch_size=3)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384
