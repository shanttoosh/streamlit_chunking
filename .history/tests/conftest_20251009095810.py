# Test Configuration
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    data = {
        'id': [1, 2, 3, 4, 5],
        'text': [
            'This is a sample text for testing.',
            'Another sample text with different content.',
            'Third sample text for chunking tests.',
            'Fourth sample text with more content.',
            'Fifth sample text for embedding tests.'
        ],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'score': [0.8, 0.6, 0.9, 0.7, 0.5]
    }
    return pd.DataFrame(data)

@pytest.fixture
def large_dataframe():
    """Create a larger DataFrame for testing"""
    data = {
        'id': range(1000),
        'text': [f'This is sample text number {i} for testing purposes.' for i in range(1000)],
        'category': [f'Category_{i % 10}' for i in range(1000)],
        'score': np.random.random(1000)
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_chunks():
    """Create mock chunks for testing"""
    return [
        "This is a sample text for testing.",
        "Another sample text with different content.",
        "Third sample text for chunking tests.",
        "Fourth sample text with more content.",
        "Fifth sample text for embedding tests."
    ]

@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing"""
    return np.random.rand(5, 384).astype(np.float32)

@pytest.fixture
def mock_metadata():
    """Create mock metadata for testing"""
    return [
        {'chunk_id': 0, 'source': 'test'},
        {'chunk_id': 1, 'source': 'test'},
        {'chunk_id': 2, 'source': 'test'},
        {'chunk_id': 3, 'source': 'test'},
        {'chunk_id': 4, 'source': 'test'}
    ]

@pytest.fixture
def test_config():
    """Create test configuration"""
    return {
        'chunk_size': 400,
        'overlap': 50,
        'model_name': 'paraphrase-MiniLM-L6-v2',
        'storage_type': 'faiss',
        'batch_size': 64
    }

@pytest.fixture(scope="session")
def test_session():
    """Session-level test setup"""
    # Create test directories
    test_dirs = ['tests/temp', 'tests/storage', 'tests/logs']
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup after session
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)
