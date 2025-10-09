# Unit Tests for Preprocessing
import pytest
import pandas as pd
import numpy as np
from src.core.preprocessing import (
    preprocess_basic,
    preprocess_auto_fast,
    preprocess_optimized_fast,
    clean_text_advanced,
    clean_column_names,
    convert_column_types
)

class TestPreprocessing:
    """Test preprocessing functions"""
    
    def test_preprocess_basic_drop_nulls(self, sample_dataframe):
        """Test basic preprocessing with null dropping"""
        # Add some null values
        df = sample_dataframe.copy()
        df.loc[1, 'text'] = None
        df.loc[3, 'score'] = None
        
        result = preprocess_basic(df, null_handling="drop")
        
        assert len(result) < len(df)
        assert result.isnull().sum().sum() == 0
    
    def test_preprocess_basic_fill_nulls(self, sample_dataframe):
        """Test basic preprocessing with null filling"""
        # Add some null values
        df = sample_dataframe.copy()
        df.loc[1, 'text'] = None
        df.loc[3, 'score'] = None
        
        result = preprocess_basic(df, null_handling="fill", fill_value="Unknown")
        
        assert len(result) == len(df)
        assert result['text'].isnull().sum() == 0
    
    def test_preprocess_auto_fast(self, sample_dataframe):
        """Test auto fast preprocessing"""
        result = preprocess_auto_fast(sample_dataframe)
        
        assert len(result) <= len(sample_dataframe)
        assert isinstance(result, pd.DataFrame)
    
    def test_preprocess_optimized_fast(self, sample_dataframe):
        """Test optimized fast preprocessing"""
        result = preprocess_optimized_fast(sample_dataframe)
        
        assert len(result) <= len(sample_dataframe)
        assert isinstance(result, pd.DataFrame)
    
    def test_clean_text_advanced(self, sample_dataframe):
        """Test advanced text cleaning"""
        text_series = sample_dataframe['text']
        
        result = clean_text_advanced(
            text_series, 
            lowercase=True, 
            remove_delimiters=True, 
            remove_whitespace=True
        )
        
        assert len(result) == len(text_series)
        assert isinstance(result, pd.Series)
    
    def test_clean_column_names(self, sample_dataframe):
        """Test column name cleaning"""
        df = sample_dataframe.copy()
        df.columns = ['ID', 'Text Column', 'Category-Type', 'Score Value']
        
        result = clean_column_names(df)
        
        assert all(col.islower() for col in result.columns)
        assert all('_' in col or col.isalnum() for col in result.columns)
    
    def test_convert_column_types(self, sample_dataframe):
        """Test column type conversion"""
        df = sample_dataframe.copy()
        column_types = {
            'id': 'string',
            'score': 'float'
        }
        
        result_df, results = convert_column_types(df, column_types)
        
        assert result_df['id'].dtype == 'object'  # string
        assert result_df['score'].dtype == 'float64'
        assert 'successful' in results
        assert len(results['successful']) > 0
    
    def test_convert_column_types_invalid(self, sample_dataframe):
        """Test column type conversion with invalid types"""
        df = sample_dataframe.copy()
        column_types = {
            'nonexistent': 'string',
            'id': 'invalid_type'
        }
        
        result_df, results = convert_column_types(df, column_types)
        
        assert 'skipped' in results
        assert 'failed' in results
        assert len(results['skipped']) > 0 or len(results['failed']) > 0
