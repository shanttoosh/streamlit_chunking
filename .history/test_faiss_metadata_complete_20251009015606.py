#!/usr/bin/env python3
"""
Complete Test of FAISS Metadata Functionality
"""
import requests
import json
import pandas as pd
import numpy as np
import time

def test_complete_faiss_metadata():
    """Test complete FAISS metadata functionality with document-based chunking"""
    print('🧪 TESTING COMPLETE FAISS METADATA FUNCTIONALITY')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Test data with clear metadata columns
    test_data = {
        'text': [
            'The system is running smoothly and efficiently in New York',
            'We are analyzing the data carefully and thoroughly in Los Angeles',
            'The machines are working better than expected in New York',
            'Processing information automatically and accurately in Chicago',
            'The results show significant improvement over time in Los Angeles'
        ],
        'score': [85.5, 92.3, 78.1, 88.7, 91.2],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
        'category': ['A', 'B', 'A', 'C', 'B'],
        'department': ['IT', 'Analytics', 'IT', 'Operations', 'Analytics']
    }
    
    df = pd.DataFrame(test_data)
    
    print('Test DataFrame:')
    print(df)
    print()
    
    # Step 1: Preprocess
    print('STEP 1: Preprocessing')
    print('-' * 30)
    try:
        files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
        response = requests.post(f"{base_url}/deep_config/preprocess", files=files)
        print(f'Preprocess status: {response.status_code}')
        if response.status_code == 200:
            result = response.json()
            print(f'Preprocess result: {result.get("status", "unknown")}')
        else:
            print(f'Preprocess error: {response.text}')
            return
    except Exception as e:
        print(f'Preprocess error: {e}')
        return
    
    print()
    
    # Step 2: Chunking with Document-based method and metadata
    print('STEP 2: Document-based Chunking with Metadata')
    print('-' * 30)
    try:
        chunk_data = {
            'chunk_method': 'document',
            'key_column': 'city',  # Group by city
            'token_limit': 1000,
            'store_metadata': 'true',
            'numeric_columns': 1,  # Include score column
            'categorical_columns': 2  # Include category and department
        }
        response = requests.post(f"{base_url}/deep_config/chunk", data=chunk_data)
        print(f'Chunking status: {response.status_code}')
        if response.status_code == 200:
            result = response.json()
            print(f'Chunks created: {result.get("total_chunks", 0)}')
            print(f'Metadata enabled: {result.get("metadata_enabled", False)}')
            
            # Show sample metadata
            metadata = result.get('metadata', [])
            if metadata:
                print('Sample metadata:')
                for i, meta in enumerate(metadata[:2]):
                    print(f'  Chunk {i+1}: {meta}')
        else:
            print(f'Chunking error: {response.text}')
            return
    except Exception as e:
        print(f'Chunking error: {e}')
        return
    
    print()
    
    # Step 3: Embedding
    print('STEP 3: Embedding')
    print('-' * 30)
    try:
        embed_data = {'model_name': 'all-MiniLM-L6-v2'}
        response = requests.post(f"{base_url}/deep_config/embed", data=embed_data)
        print(f'Embedding status: {response.status_code}')
        if response.status_code == 200:
            result = response.json()
            print(f'Embedding result: {result.get("status", "unknown")}')
        else:
            print(f'Embedding error: {response.text}')
            return
    except Exception as e:
        print(f'Embedding error: {e}')
        return
    
    print()
    
    # Step 4: Store in FAISS
    print('STEP 4: Store in FAISS')
    print('-' * 30)
    try:
        store_data = {'storage_type': 'faiss'}
        response = requests.post(f"{base_url}/deep_config/store", data=store_data)
        print(f'Storage status: {response.status_code}')
        if response.status_code == 200:
            result = response.json()
            print(f'Storage result: {result.get("status", "unknown")}')
            print(f'Storage type: {result.get("storage_type", "unknown")}')
            print(f'Total vectors: {result.get("total_vectors", 0)}')
        else:
            print(f'Storage error: {response.text}')
            return
    except Exception as e:
        print(f'Storage error: {e}')
        return
    
    print()
    
    # Step 5: Test metadata filtering with various filters
    print('STEP 5: Test Metadata Filtering')
    print('-' * 30)
    
    # Test different queries with metadata filters
    test_queries = [
        {
            'name': 'No filter',
            'query': 'system performance',
            'k': 5,
            'metadata_filter': '{}'
        },
        {
            'name': 'Filter by city (NYC)',
            'query': 'system performance',
            'k': 5,
            'metadata_filter': '{"city_mode": "NYC"}'
        },
        {
            'name': 'Filter by category (A)',
            'query': 'data analysis',
            'k': 3,
            'metadata_filter': '{"category_mode": "A"}'
        },
        {
            'name': 'Filter by department (IT)',
            'query': 'machines working',
            'k': 3,
            'metadata_filter': '{"department_mode": "IT"}'
        },
        {
            'name': 'Filter by score range (high scores)',
            'query': 'results improvement',
            'k': 3,
            'metadata_filter': '{"score_mean": 90.0}'  # This might not work as expected
        },
        {
            'name': 'Multiple filters (NYC + IT)',
            'query': 'system efficiency',
            'k': 3,
            'metadata_filter': '{"city_mode": "NYC", "department_mode": "IT"}'
        }
    ]
    
    for test in test_queries:
        print(f'\\n{test["name"]}:')
        print(f'  Query: {test["query"]}')
        print(f'  Filter: {test["metadata_filter"]}')
        
        try:
            query_data = {
                'query': test['query'],
                'k': test['k'],
                'metadata_filter': test['metadata_filter']
            }
            response = requests.post(f"{base_url}/retrieve_with_metadata", data=query_data)
            print(f'  Status: {response.status_code}')
            
            if response.status_code == 200:
                result = response.json()
                print(f'  Results: {result.get("total_results", 0)}')
                print(f'  Filter applied: {result.get("metadata_filter_applied", False)}')
                
                # Show top results with metadata
                results = result.get('results', [])
                for j, res in enumerate(results[:2]):  # Show top 2
                    print(f'    {j+1}. Similarity: {res.get("similarity", 0):.3f}')
                    print(f'       Metadata: {res.get("metadata", {})}')
            else:
                print(f'  Error: {response.text}')
                
        except Exception as e:
            print(f'  Error: {e}')
    
    print('=' * 60)

if __name__ == "__main__":
    print('Starting complete FAISS metadata test...')
    print('Make sure the FastAPI server is running on localhost:8000')
    print()
    
    try:
        test_complete_faiss_metadata()
        print('✅ Complete FAISS metadata testing completed!')
    except Exception as e:
        print(f'❌ Complete FAISS metadata testing failed: {e}')
        print('Make sure the server is running: uvicorn main:app --reload')
