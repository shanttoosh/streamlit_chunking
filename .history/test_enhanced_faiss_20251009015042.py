#!/usr/bin/env python3
"""
Test Enhanced FAISS Metadata Functionality
"""
import pandas as pd
import numpy as np
import json
from backend import (
    store_faiss_enhanced, 
    query_faiss_with_metadata, 
    create_metadata_index,
    apply_metadata_filter
)

def test_enhanced_faiss_metadata():
    """Test enhanced FAISS metadata functionality"""
    print('🧪 TESTING ENHANCED FAISS METADATA FUNCTIONALITY')
    print('=' * 60)
    
    # Create test data
    chunks = [
        "The system is running smoothly and efficiently",
        "We are analyzing the data carefully and thoroughly", 
        "The machines are working better than expected",
        "Processing information automatically and accurately",
        "The results show significant improvement over time"
    ]
    
    embeddings = np.random.rand(5, 384)  # 5 chunks, 384-dimensional
    
    metadata = [
        {'chunk_id': '0', 'key_value': '1', 'score_mean': 85.5, 'city_mode': 'NYC', 'category': 'A'},
        {'chunk_id': '1', 'key_value': '2', 'score_mean': 92.3, 'city_mode': 'LA', 'category': 'B'},
        {'chunk_id': '2', 'key_value': '1', 'score_mean': 78.1, 'city_mode': 'NYC', 'category': 'A'},
        {'chunk_id': '3', 'key_value': '3', 'score_mean': 88.7, 'city_mode': 'Chicago', 'category': 'C'},
        {'chunk_id': '4', 'key_value': '2', 'score_mean': 91.2, 'city_mode': 'LA', 'category': 'B'}
    ]
    
    print('Test data:')
    print(f'Chunks: {len(chunks)}')
    print(f'Embeddings: {embeddings.shape}')
    print(f'Metadata: {len(metadata)} entries')
    print()
    
    # Test 1: Enhanced FAISS storage
    print('TEST 1: Enhanced FAISS Storage')
    print('-' * 40)
    try:
        faiss_result = store_faiss_enhanced(chunks, embeddings, metadata)
        print('✅ Enhanced FAISS storage successful')
        print(f'Result type: {faiss_result["type"]}')
        print(f'Metadata index created: {"metadata_index" in faiss_result}')
        
        # Check metadata index structure
        if "metadata_index" in faiss_result:
            metadata_index = faiss_result["metadata_index"]
            print(f'Metadata index keys: {list(metadata_index.keys())}')
            print(f'Sample index for "city_mode": {list(metadata_index.get("city_mode", {}).keys())}')
        
    except Exception as e:
        print(f'❌ Enhanced FAISS storage failed: {e}')
        return
    
    print()
    
    # Test 2: Metadata filtering
    print('TEST 2: Metadata Filtering')
    print('-' * 40)
    
    # Test different filters
    filters = [
        {},  # No filter
        {'city_mode': 'NYC'},  # Filter by city
        {'category': 'A'},  # Filter by category
        {'key_value': '1'},  # Filter by key value
        {'city_mode': 'NYC', 'category': 'A'},  # Multiple filters
        {'score_mean': 85.5},  # Filter by numeric value
        {'city_mode': 'Miami'}  # No matches
    ]
    
    for i, filter_dict in enumerate(filters):
        print(f'Filter {i+1}: {filter_dict}')
        
        try:
            # Test metadata index filtering
            matching_indices = apply_metadata_filter(faiss_result["metadata_index"], filter_dict)
            print(f'  Matching indices: {matching_indices}')
            
            # Test query with metadata filter
            query_embedding = np.random.rand(1, 384).astype("float32")
            results = query_faiss_with_metadata(
                faiss_result["index"], 
                faiss_result["data"], 
                query_embedding, 
                k=3, 
                metadata_filter=filter_dict
            )
            
            print(f'  Query results: {len(results)}')
            if results:
                print(f'  Top result metadata: {results[0].get("metadata", {})}')
            
        except Exception as e:
            print(f'  Error: {e}')
        
        print()
    
    print('=' * 60)

def test_metadata_index_performance():
    """Test metadata index performance"""
    print('TEST 3: Metadata Index Performance')
    print('-' * 40)
    
    import time
    
    # Create larger dataset
    n_chunks = 1000
    chunks = [f"This is chunk {i} with some text content" for i in range(n_chunks)]
    embeddings = np.random.rand(n_chunks, 384)
    
    # Create metadata with various patterns
    metadata = []
    for i in range(n_chunks):
        metadata.append({
            'chunk_id': str(i),
            'key_value': str(i % 10),  # 10 different key values
            'score_mean': 70 + (i % 30),  # 30 different scores
            'city_mode': ['NYC', 'LA', 'Chicago', 'Miami', 'Seattle'][i % 5],  # 5 cities
            'category': ['A', 'B', 'C', 'D', 'E'][i % 5]  # 5 categories
        })
    
    print(f'Large dataset: {n_chunks} chunks with metadata')
    
    # Test metadata index creation performance
    start_time = time.time()
    metadata_index = create_metadata_index(metadata)
    index_time = time.time() - start_time
    
    print(f'Metadata index creation: {index_time:.3f} seconds')
    print(f'Index size: {len(metadata_index)} keys')
    print(f'Sample index structure:')
    for key, values in list(metadata_index.items())[:3]:
        print(f'  {key}: {len(values)} unique values')
    
    # Test filtering performance
    filter_tests = [
        {'city_mode': 'NYC'},
        {'category': 'A'},
        {'key_value': '1'},
        {'city_mode': 'NYC', 'category': 'A'}
    ]
    
    print('\\nFiltering performance:')
    for filter_dict in filter_tests:
        start_time = time.time()
        matching_indices = apply_metadata_filter(metadata_index, filter_dict)
        filter_time = time.time() - start_time
        
        print(f'  Filter {filter_dict}: {len(matching_indices)} matches in {filter_time:.6f}s')
    
    print('=' * 60)

def test_api_simulation():
    """Simulate API calls with metadata filtering"""
    print('TEST 4: API Simulation')
    print('-' * 40)
    
    # Simulate the API call flow
    chunks = [
        "The system is running smoothly",
        "We are analyzing data carefully",
        "The machines are working efficiently"
    ]
    
    embeddings = np.random.rand(3, 384)
    metadata = [
        {'chunk_id': '0', 'key_value': '1', 'city_mode': 'NYC'},
        {'chunk_id': '1', 'key_value': '2', 'city_mode': 'LA'},
        {'chunk_id': '2', 'key_value': '1', 'city_mode': 'NYC'}
    ]
    
    # Store in FAISS
    faiss_result = store_faiss_enhanced(chunks, embeddings, metadata)
    
    # Simulate API calls
    api_tests = [
        {
            'query': 'system performance',
            'k': 3,
            'metadata_filter': '{}'  # No filter
        },
        {
            'query': 'system performance', 
            'k': 3,
            'metadata_filter': '{"city_mode": "NYC"}'  # Filter by city
        },
        {
            'query': 'data analysis',
            'k': 2,
            'metadata_filter': '{"key_value": "1"}'  # Filter by key
        }
    ]
    
    print('Simulating API calls:')
    for i, test in enumerate(api_tests):
        print(f'\\nAPI Call {i+1}:')
        print(f'  Query: {test["query"]}')
        print(f'  K: {test["k"]}')
        print(f'  Filter: {test["metadata_filter"]}')
        
        try:
            # Parse filter
            filter_dict = json.loads(test["metadata_filter"])
            
            # Simulate query
            query_embedding = np.random.rand(1, 384).astype("float32")
            results = query_faiss_with_metadata(
                faiss_result["index"],
                faiss_result["data"], 
                query_embedding,
                test["k"],
                filter_dict
            )
            
            print(f'  Results: {len(results)}')
            for j, result in enumerate(results):
                print(f'    {j+1}. Similarity: {result["similarity"]:.3f}, Metadata: {result["metadata"]}')
                
        except Exception as e:
            print(f'  Error: {e}')
    
    print('=' * 60)

if __name__ == "__main__":
    test_enhanced_faiss_metadata()
    test_metadata_index_performance()
    test_api_simulation()
    print('✅ Enhanced FAISS metadata testing completed!')
