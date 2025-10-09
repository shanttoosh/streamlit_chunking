#!/usr/bin/env python3
"""
Test Metadata Handling System
"""
import pandas as pd
import numpy as np
from backend import document_based_chunking_enhanced, store_chroma_enhanced

def test_metadata_generation():
    """Test metadata generation during chunking"""
    print('🧪 TESTING METADATA GENERATION')
    print('=' * 60)
    
    # Create test dataset
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'David', 'Alice', 'Eve'],
        'age': [25, 30, 25, 35, 30, 28, 25, 32],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'Miami', 'NYC', 'Seattle'],
        'score': [85.5, 92.3, 78.1, 88.7, 91.2, 79.8, 86.4, 93.1],
        'category': ['A', 'B', 'A', 'C', 'B', 'D', 'A', 'E']
    }
    
    df = pd.DataFrame(data)
    print('Test dataset:')
    print(df)
    print()
    
    # Test document-based chunking with metadata
    print('Testing document-based chunking with metadata:')
    print('-' * 50)
    
    try:
        chunks, metadata = document_based_chunking_enhanced(df, key_column='name', token_limit=100)
        
        print(f'Generated {len(chunks)} chunks with metadata')
        print()
        
        # Show metadata for each chunk
        for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
            print(f'Chunk {i+1}:')
            print(f'  Metadata: {meta}')
            print(f'  Chunk preview: {chunk[:100]}...')
            print()
            
    except Exception as e:
        print(f'Error: {e}')
    
    print('=' * 60)

def test_metadata_structure():
    """Test metadata structure and content"""
    print('TESTING METADATA STRUCTURE')
    print('=' * 60)
    
    # Create simple test data
    data = {
        'user_id': [1, 2, 3, 4, 5],
        'product': ['A', 'B', 'A', 'C', 'B'],
        'price': [10.0, 20.0, 10.0, 30.0, 20.0],
        'rating': [4.5, 3.8, 4.5, 4.2, 3.8]
    }
    
    df = pd.DataFrame(data)
    print('Test data:')
    print(df)
    print()
    
    # Test different chunking methods and their metadata
    methods = [
        ('name', 'user_id'),
        ('product', 'product'),
        ('rating', 'rating')
    ]
    
    for method_name, key_column in methods:
        print(f'Testing chunking by {key_column}:')
        print('-' * 30)
        
        try:
            chunks, metadata = document_based_chunking_enhanced(df, key_column=key_column, token_limit=50)
            
            print(f'Chunks: {len(chunks)}')
            print(f'Metadata entries: {len(metadata)}')
            print()
            
            # Analyze metadata structure
            if metadata:
                sample_meta = metadata[0]
                print('Sample metadata structure:')
                for key, value in sample_meta.items():
                    print(f'  {key}: {value} ({type(value).__name__})')
                print()
                
                # Show all metadata
                print('All metadata:')
                for i, meta in enumerate(metadata):
                    print(f'  Chunk {i+1}: {meta}')
                print()
                
        except Exception as e:
            print(f'Error: {e}')
        
        print('=' * 60)
        print()

def test_metadata_storage():
    """Test metadata storage in ChromaDB"""
    print('TESTING METADATA STORAGE')
    print('=' * 60)
    
    # Create test chunks and metadata
    chunks = [
        "This is the first chunk with some text content",
        "This is the second chunk with different content",
        "This is the third chunk with more text content"
    ]
    
    # Create sample embeddings (dummy)
    embeddings = np.random.rand(3, 384)  # 3 chunks, 384-dimensional embeddings
    
    # Create metadata
    metadata = [
        {
            'chunk_index': 0,
            'key_column': 'user_id',
            'key_value': '1',
            'chunking_method': 'document_based',
            'token_count': 10,
            'group_size': 2
        },
        {
            'chunk_index': 1,
            'key_column': 'user_id', 
            'key_value': '2',
            'chunking_method': 'document_based',
            'token_count': 12,
            'group_size': 1
        },
        {
            'chunk_index': 2,
            'key_column': 'user_id',
            'key_value': '3', 
            'chunking_method': 'document_based',
            'token_count': 11,
            'group_size': 1
        }
    ]
    
    print('Test chunks:')
    for i, chunk in enumerate(chunks):
        print(f'  Chunk {i+1}: {chunk}')
    print()
    
    print('Test metadata:')
    for i, meta in enumerate(metadata):
        print(f'  Chunk {i+1}: {meta}')
    print()
    
    # Test storage (this will fail if ChromaDB not available, but we can see the structure)
    print('Testing metadata storage in ChromaDB:')
    try:
        result = store_chroma_enhanced(chunks, embeddings, "test_collection", metadata)
        print('Storage successful!')
        print(f'Result: {result}')
    except Exception as e:
        print(f'Storage test (expected if ChromaDB not available): {e}')
    
    print('=' * 60)

def test_metadata_ui_simulation():
    """Simulate metadata UI selection logic"""
    print('TESTING METADATA UI SIMULATION')
    print('=' * 60)
    
    # Create test dataset
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['NYC', 'LA', 'Chicago', 'Miami', 'Seattle'],
        'score': [85.5, 92.3, 78.1, 88.7, 91.2],
        'category': ['A', 'B', 'C', 'D', 'E'],
        'description': ['Good', 'Excellent', 'Average', 'Poor', 'Great']
    }
    
    df = pd.DataFrame(data)
    print('Test dataset:')
    print(df)
    print()
    
    # Simulate UI logic for metadata selection
    print('Simulating metadata selection UI logic:')
    print('-' * 40)
    
    # Numeric columns
    numeric_candidates = df.select_dtypes(include=['number']).columns.tolist()
    print(f'Numeric candidates: {numeric_candidates}')
    
    # Categorical columns (low cardinality)
    max_categorical_cardinality = 50
    raw_categorical = df.select_dtypes(include=['object']).columns.tolist()
    categorical_candidates = [c for c in raw_categorical if df[c].nunique(dropna=True) <= max_categorical_cardinality]
    print(f'Categorical candidates: {categorical_candidates}')
    print()
    
    # Ranking logic (simulate UI ranking)
    def _num_rank(col):
        try:
            var = float(pd.to_numeric(df[col], errors='coerce').var())
        except Exception:
            var = 0.0
        miss = float(pd.to_numeric(df[col], errors='coerce').isna().mean())
        return (-var, miss)  # Higher variance, lower missing = better
    
    ranked_numeric = sorted(numeric_candidates, key=_num_rank)
    print(f'Ranked numeric columns: {ranked_numeric}')
    
    def _cat_rank(col):
        s = df[col]
        miss = float(s.isna().mean())
        uniq = int(s.nunique(dropna=True))
        return (miss, uniq)  # Lower missing, lower cardinality = better
    
    ranked_categorical = sorted(categorical_candidates, key=_cat_rank)
    print(f'Ranked categorical columns: {ranked_categorical}')
    print()
    
    # Simulate user selection
    max_numeric_cap = min(10, len(numeric_candidates))
    max_categorical_cap = min(5, len(categorical_candidates))
    
    selected_numeric = ranked_numeric[:max_numeric_cap]
    selected_categorical = ranked_categorical[:min(2, max_categorical_cap)]
    
    print(f'Selected numeric columns: {selected_numeric}')
    print(f'Selected categorical columns: {selected_categorical}')
    print()
    
    # Show what would be stored as metadata
    print('Metadata that would be stored:')
    for col in selected_numeric:
        print(f'  Numeric {col}: min/mean/max per chunk')
    for col in selected_categorical:
        print(f'  Categorical {col}: mode per chunk')
    
    print('=' * 60)

if __name__ == "__main__":
    test_metadata_generation()
    test_metadata_structure()
    test_metadata_storage()
    test_metadata_ui_simulation()
    print('✅ All metadata handling tests completed!')
