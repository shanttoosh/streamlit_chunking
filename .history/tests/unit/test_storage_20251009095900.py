# Unit Tests for Storage
import pytest
import numpy as np
from src.core.storage import (
    store_chroma,
    store_faiss,
    store_faiss_with_metric,
    store_chroma_with_metric
)

class TestStorage:
    """Test storage functions"""
    
    def test_store_faiss(self, mock_embeddings):
        """Test FAISS storage"""
        store_info = store_faiss(mock_embeddings)
        
        assert isinstance(store_info, dict)
        assert store_info["type"] == "faiss"
        assert "index" in store_info
        assert store_info["index"] is not None
    
    def test_store_faiss_with_metric_cosine(self, mock_embeddings):
        """Test FAISS storage with cosine metric"""
        store_info = store_faiss_with_metric(mock_embeddings, metric="cosine")
        
        assert isinstance(store_info, dict)
        assert store_info["type"] == "faiss"
        assert store_info["metric"] == "cosine"
        assert "index" in store_info
        assert store_info["index"] is not None
    
    def test_store_faiss_with_metric_dot(self, mock_embeddings):
        """Test FAISS storage with dot product metric"""
        store_info = store_faiss_with_metric(mock_embeddings, metric="dot")
        
        assert isinstance(store_info, dict)
        assert store_info["type"] == "faiss"
        assert store_info["metric"] == "dot"
        assert "index" in store_info
        assert store_info["index"] is not None
    
    def test_store_faiss_with_metric_euclidean(self, mock_embeddings):
        """Test FAISS storage with euclidean metric"""
        store_info = store_faiss_with_metric(mock_embeddings, metric="euclidean")
        
        assert isinstance(store_info, dict)
        assert store_info["type"] == "faiss"
        assert store_info["metric"] == "euclidean"
        assert "index" in store_info
        assert store_info["index"] is not None
    
    def test_store_chroma(self, mock_chunks, mock_embeddings):
        """Test ChromaDB storage"""
        store_info = store_chroma(mock_chunks, mock_embeddings)
        
        assert isinstance(store_info, dict)
        assert store_info["type"] == "chroma"
        assert "collection" in store_info
        assert "collection_name" in store_info
        assert store_info["collection"] is not None
    
    def test_store_chroma_with_metric_cosine(self, mock_chunks, mock_embeddings):
        """Test ChromaDB storage with cosine metric"""
        store_info = store_chroma_with_metric(
            mock_chunks, 
            mock_embeddings, 
            metric="cosine"
        )
        
        assert isinstance(store_info, dict)
        assert store_info["type"] == "chroma"
        assert store_info["space"] == "cosine"
        assert "collection" in store_info
        assert store_info["collection"] is not None
    
    def test_store_chroma_with_metric_l2(self, mock_chunks, mock_embeddings):
        """Test ChromaDB storage with L2 metric"""
        store_info = store_chroma_with_metric(
            mock_chunks, 
            mock_embeddings, 
            metric="euclidean"
        )
        
        assert isinstance(store_info, dict)
        assert store_info["type"] == "chroma"
        assert store_info["space"] == "l2"
        assert "collection" in store_info
        assert store_info["collection"] is not None
    
    def test_store_chroma_with_metric_ip(self, mock_chunks, mock_embeddings):
        """Test ChromaDB storage with inner product metric"""
        store_info = store_chroma_with_metric(
            mock_chunks, 
            mock_embeddings, 
            metric="dot"
        )
        
        assert isinstance(store_info, dict)
        assert store_info["type"] == "chroma"
        assert store_info["space"] == "ip"
        assert "collection" in store_info
        assert store_info["collection"] is not None
    
    def test_store_faiss_empty_embeddings(self):
        """Test FAISS storage with empty embeddings"""
        empty_embeddings = np.array([]).reshape(0, 384)
        
        with pytest.raises(Exception):
            store_faiss(empty_embeddings)
    
    def test_store_chroma_empty_chunks(self, mock_embeddings):
        """Test ChromaDB storage with empty chunks"""
        empty_chunks = []
        
        with pytest.raises(Exception):
            store_chroma(empty_chunks, mock_embeddings)
    
    def test_store_chroma_mismatched_lengths(self, mock_chunks, mock_embeddings):
        """Test ChromaDB storage with mismatched chunk and embedding lengths"""
        # Create embeddings with different length than chunks
        mismatched_embeddings = np.random.rand(len(mock_chunks) + 1, 384).astype(np.float32)
        
        with pytest.raises(Exception):
            store_chroma(mock_chunks, mismatched_embeddings)
    
    def test_store_faiss_large_embeddings(self):
        """Test FAISS storage with large embeddings"""
        # Create large embeddings
        large_embeddings = np.random.rand(1000, 384).astype(np.float32)
        
        store_info = store_faiss(large_embeddings)
        
        assert isinstance(store_info, dict)
        assert store_info["type"] == "faiss"
        assert "index" in store_info
        assert store_info["index"] is not None
