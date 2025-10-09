# Unit Tests for Chunking
import pytest
import pandas as pd
from src.core.chunking import (
    chunk_fixed,
    chunk_recursive_keyvalue,
    chunk_semantic_cluster,
    document_based_chunking
)

class TestChunking:
    """Test chunking functions"""
    
    def test_chunk_fixed(self, sample_dataframe):
        """Test fixed chunking"""
        chunks = chunk_fixed(sample_dataframe, chunk_size=100, overlap=10)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_recursive_keyvalue(self, sample_dataframe):
        """Test recursive key-value chunking"""
        chunks = chunk_recursive_keyvalue(sample_dataframe, chunk_size=100, overlap=10)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_semantic_cluster(self, sample_dataframe):
        """Test semantic clustering chunking"""
        chunks = chunk_semantic_cluster(sample_dataframe, n_clusters=3)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_document_based_chunking(self, sample_dataframe):
        """Test document-based chunking"""
        chunks, metadata = document_based_chunking(
            sample_dataframe, 
            key_column='category',
            token_limit=1000,
            preserve_headers=True
        )
        
        assert isinstance(chunks, list)
        assert isinstance(metadata, list)
        assert len(chunks) > 0
        assert len(chunks) == len(metadata)
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(isinstance(meta, dict) for meta in metadata)
    
    def test_document_based_chunking_invalid_column(self, sample_dataframe):
        """Test document-based chunking with invalid column"""
        with pytest.raises(ValueError):
            document_based_chunking(
                sample_dataframe,
                key_column='nonexistent',
                token_limit=1000
            )
    
    def test_chunk_fixed_empty_dataframe(self):
        """Test fixed chunking with empty DataFrame"""
        empty_df = pd.DataFrame()
        chunks = chunk_fixed(empty_df, chunk_size=100, overlap=10)
        
        assert isinstance(chunks, list)
        assert len(chunks) == 0
    
    def test_chunk_semantic_cluster_single_cluster(self, sample_dataframe):
        """Test semantic clustering with single cluster"""
        chunks = chunk_semantic_cluster(sample_dataframe, n_clusters=1)
        
        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert isinstance(chunks[0], str)
    
    def test_document_based_chunking_preserve_headers(self, sample_dataframe):
        """Test document-based chunking with header preservation"""
        chunks, metadata = document_based_chunking(
            sample_dataframe,
            key_column='category',
            token_limit=1000,
            preserve_headers=True
        )
        
        # Check that headers are preserved in chunks
        assert any('HEADERS:' in chunk for chunk in chunks)
    
    def test_document_based_chunking_no_headers(self, sample_dataframe):
        """Test document-based chunking without header preservation"""
        chunks, metadata = document_based_chunking(
            sample_dataframe,
            key_column='category',
            token_limit=1000,
            preserve_headers=False
        )
        
        # Check that headers are not preserved in chunks
        assert not any('HEADERS:' in chunk for chunk in chunks)
