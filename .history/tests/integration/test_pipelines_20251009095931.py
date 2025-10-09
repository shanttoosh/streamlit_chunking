# Integration Tests for Pipelines
import pytest
import pandas as pd
import numpy as np
from src.core.pipelines import (
    run_fast_pipeline,
    run_config1_pipeline,
    run_deep_config_pipeline,
    save_state,
    load_state
)

class TestPipelines:
    """Test processing pipelines"""
    
    def test_run_fast_pipeline(self, sample_dataframe):
        """Test fast pipeline execution"""
        file_info = {"filename": "test.csv", "size": 1000}
        
        result = run_fast_pipeline(
            df=sample_dataframe,
            file_info=file_info,
            use_openai=False,
            use_turbo=True,
            batch_size=256
        )
        
        assert isinstance(result, dict)
        assert "rows" in result
        assert "chunks" in result
        assert "stored" in result
        assert "embedding_model" in result
        assert "retrieval_ready" in result
        assert result["retrieval_ready"] == True
        assert result["rows"] == len(sample_dataframe)
    
    def test_run_config1_pipeline(self, sample_dataframe):
        """Test config1 pipeline execution"""
        file_info = {"filename": "test.csv", "size": 1000}
        
        result = run_config1_pipeline(
            df=sample_dataframe,
            chunk_method="recursive",
            chunk_size=400,
            overlap=50,
            model_choice="paraphrase-MiniLM-L6-v2",
            storage_choice="faiss",
            file_info=file_info,
            use_openai=False,
            use_turbo=False,
            batch_size=256,
            retrieval_metric="cosine"
        )
        
        assert isinstance(result, dict)
        assert "rows" in result
        assert "chunks" in result
        assert "stored" in result
        assert "embedding_model" in result
        assert "retrieval_ready" in result
        assert result["retrieval_ready"] == True
        assert result["rows"] == len(sample_dataframe)
    
    def test_run_deep_config_pipeline(self, sample_dataframe):
        """Test deep config pipeline execution"""
        file_info = {"filename": "test.csv", "size": 1000}
        
        config_dict = {
            "preprocessing": {
                "remove_duplicates": True,
                "clean_headers": True
            },
            "chunking": {
                "method": "fixed",
                "chunk_size": 400,
                "overlap": 50
            },
            "embedding": {
                "model_name": "paraphrase-MiniLM-L6-v2",
                "batch_size": 64,
                "use_parallel": True
            },
            "storage": {
                "type": "chroma",
                "collection_name": "test_collection"
            }
        }
        
        result = run_deep_config_pipeline(
            df=sample_dataframe,
            config_dict=config_dict,
            file_info=file_info
        )
        
        assert isinstance(result, dict)
        assert "rows" in result
        assert "chunks" in result
        assert "stored" in result
        assert "embedding_model" in result
        assert "retrieval_ready" in result
        assert "enhanced_pipeline" in result
        assert result["retrieval_ready"] == True
        assert result["enhanced_pipeline"] == True
        assert result["rows"] == len(sample_dataframe)
    
    def test_run_fast_pipeline_with_openai(self, sample_dataframe):
        """Test fast pipeline with OpenAI (will fallback to local)"""
        file_info = {"filename": "test.csv", "size": 1000}
        
        result = run_fast_pipeline(
            df=sample_dataframe,
            file_info=file_info,
            use_openai=True,
            openai_api_key=None,  # No API key, should fallback
            use_turbo=False,
            batch_size=256
        )
        
        assert isinstance(result, dict)
        assert "rows" in result
        assert "chunks" in result
        assert "stored" in result
        assert "embedding_model" in result
        assert "retrieval_ready" in result
        assert result["retrieval_ready"] == True
    
    def test_run_config1_pipeline_different_methods(self, sample_dataframe):
        """Test config1 pipeline with different chunking methods"""
        file_info = {"filename": "test.csv", "size": 1000}
        
        methods = ["fixed", "recursive", "semantic", "document"]
        
        for method in methods:
            result = run_config1_pipeline(
                df=sample_dataframe,
                chunk_method=method,
                chunk_size=400,
                overlap=50,
                model_choice="paraphrase-MiniLM-L6-v2",
                storage_choice="faiss",
                file_info=file_info,
                use_openai=False,
                use_turbo=False,
                batch_size=256,
                retrieval_metric="cosine"
            )
            
            assert isinstance(result, dict)
            assert "rows" in result
            assert "chunks" in result
            assert "stored" in result
            assert "embedding_model" in result
            assert "retrieval_ready" in result
            assert result["retrieval_ready"] == True
    
    def test_run_config1_pipeline_different_storage(self, sample_dataframe):
        """Test config1 pipeline with different storage backends"""
        file_info = {"filename": "test.csv", "size": 1000}
        
        storage_options = ["faiss", "chroma"]
        
        for storage in storage_options:
            result = run_config1_pipeline(
                df=sample_dataframe,
                chunk_method="recursive",
                chunk_size=400,
                overlap=50,
                model_choice="paraphrase-MiniLM-L6-v2",
                storage_choice=storage,
                file_info=file_info,
                use_openai=False,
                use_turbo=False,
                batch_size=256,
                retrieval_metric="cosine"
            )
            
            assert isinstance(result, dict)
            assert "rows" in result
            assert "chunks" in result
            assert "stored" in result
            assert "embedding_model" in result
            assert "retrieval_ready" in result
            assert result["retrieval_ready"] == True
            assert result["stored"] == storage
    
    def test_run_deep_config_pipeline_different_chunking(self, sample_dataframe):
        """Test deep config pipeline with different chunking methods"""
        file_info = {"filename": "test.csv", "size": 1000}
        
        chunking_methods = ["fixed", "recursive", "semantic", "document"]
        
        for method in chunking_methods:
            config_dict = {
                "preprocessing": {},
                "chunking": {
                    "method": method,
                    "chunk_size": 400,
                    "overlap": 50
                },
                "embedding": {
                    "model_name": "paraphrase-MiniLM-L6-v2",
                    "batch_size": 64
                },
                "storage": {
                    "type": "faiss",
                    "collection_name": f"test_{method}"
                }
            }
            
            result = run_deep_config_pipeline(
                df=sample_dataframe,
                config_dict=config_dict,
                file_info=file_info
            )
            
            assert isinstance(result, dict)
            assert "rows" in result
            assert "chunks" in result
            assert "stored" in result
            assert "embedding_model" in result
            assert "retrieval_ready" in result
            assert "enhanced_pipeline" in result
            assert result["retrieval_ready"] == True
            assert result["enhanced_pipeline"] == True
    
    def test_state_management(self, sample_dataframe):
        """Test state save and load functionality"""
        file_info = {"filename": "test.csv", "size": 1000}
        
        # Run pipeline to set state
        result = run_fast_pipeline(
            df=sample_dataframe,
            file_info=file_info,
            use_openai=False,
            use_turbo=True,
            batch_size=256
        )
        
        # Save state
        save_state()
        
        # Load state
        loaded = load_state()
        
        # State should be loaded successfully
        assert loaded == True
    
    def test_pipeline_with_empty_dataframe(self):
        """Test pipeline with empty DataFrame"""
        empty_df = pd.DataFrame()
        file_info = {"filename": "empty.csv", "size": 0}
        
        result = run_fast_pipeline(
            df=empty_df,
            file_info=file_info,
            use_openai=False,
            use_turbo=True,
            batch_size=256
        )
        
        assert isinstance(result, dict)
        assert "rows" in result
        assert "chunks" in result
        assert "stored" in result
        assert "embedding_model" in result
        assert "retrieval_ready" in result
        assert result["rows"] == 0
    
    def test_pipeline_with_large_dataframe(self, large_dataframe):
        """Test pipeline with large DataFrame"""
        file_info = {"filename": "large.csv", "size": 100000}
        
        result = run_fast_pipeline(
            df=large_dataframe,
            file_info=file_info,
            use_openai=False,
            use_turbo=True,
            batch_size=256
        )
        
        assert isinstance(result, dict)
        assert "rows" in result
        assert "chunks" in result
        assert "stored" in result
        assert "embedding_model" in result
        assert "retrieval_ready" in result
        assert result["retrieval_ready"] == True
        assert result["rows"] == len(large_dataframe)
