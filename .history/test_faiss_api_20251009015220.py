#!/usr/bin/env python3
"""
Test FAISS API Endpoint
"""
import requests
import json
import pandas as pd
import numpy as np
import time

def test_faiss_api():
    """Test the FAISS API endpoint"""
    print('🧪 TESTING FAISS API ENDPOINT')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Test data
    test_data = {
        'text': [
            'The system is running smoothly and efficiently',
            'We are analyzing the data carefully and thoroughly',
            'The machines are working better than expected',
            'Processing information automatically and accurately',
            'The results show significant improvement over time'
        ],
        'score': [85.5, 92.3, 78.1, 88.7, 91.2],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA'],
        'category': ['A', 'B', 'A', 'C', 'B']
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
    except Exception as e:
        print(f'Preprocess error: {e}')
    
    print()
    
    # Step 2: Chunking
    print('STEP 2: Chunking')
    print('-' * 30)
    try:
        chunk_data = {
            'chunking_method': 'document_based',
            'chunk_size': 100,
            'chunk_overlap': 20,
            'text_column': 'text',
            'key_column': 'score',
            'store_metadata': 'true',
            'numeric_columns': '1',
            'categorical_columns': '1'
        }
        response = requests.post(f"{base_url}/deep_config/chunk", data=chunk_data)
        print(f'Chunking status: {response.status_code}')
        if response.status_code == 200:
            result = response.json()
            print(f'Chunks created: {result.get("total_chunks", 0)}')
            print(f'Metadata enabled: {result.get("metadata_enabled", False)}')
        else:
            print(f'Chunking error: {response.text}')
    except Exception as e:
        print(f'Chunking error: {e}')
    
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
    except Exception as e:
        print(f'Embedding error: {e}')
    
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
    except Exception as e:
        print(f'Storage error: {e}')
    
    print()
    
    # Step 5: Test metadata filtering
    print('STEP 5: Test Metadata Filtering')
    print('-' * 30)
    
    # Test different queries with metadata filters
    test_queries = [
        {
            'query': 'system performance',
            'k': 3,
            'metadata_filter': '{}'  # No filter
        },
        {
            'query': 'system performance',
            'k': 3,
            'metadata_filter': '{"city": "NYC"}'  # Filter by city
        },
        {
            'query': 'data analysis',
            'k': 2,
            'metadata_filter': '{"category": "A"}'  # Filter by category
        },
        {
            'query': 'results improvement',
            'k': 2,
            'metadata_filter': '{"city": "LA", "category": "B"}'  # Multiple filters
        }
    ]
    
    for i, test in enumerate(test_queries):
        print(f'\\nQuery {i+1}: {test["query"]}')
        print(f'Filter: {test["metadata_filter"]}')
        
        try:
            query_data = {
                'query': test['query'],
                'k': test['k'],
                'metadata_filter': test['metadata_filter']
            }
            response = requests.post(f"{base_url}/retrieve_with_metadata", data=query_data)
            print(f'Query status: {response.status_code}')
            
            if response.status_code == 200:
                result = response.json()
                print(f'Results: {result.get("total_results", 0)}')
                print(f'Filter applied: {result.get("metadata_filter_applied", False)}')
                print(f'Store type: {result.get("store_type", "unknown")}')
                
                # Show top results
                results = result.get('results', [])
                for j, res in enumerate(results[:2]):  # Show top 2
                    print(f'  {j+1}. Similarity: {res.get("similarity", 0):.3f}')
                    print(f'     Metadata: {res.get("metadata", {})}')
            else:
                print(f'Query error: {response.text}')
                
        except Exception as e:
            print(f'Query error: {e}')
    
    print('=' * 60)

if __name__ == "__main__":
    print('Starting FAISS API test...')
    print('Make sure the FastAPI server is running on localhost:8000')
    print()
    
    try:
        test_faiss_api()
        print('✅ FAISS API testing completed!')
    except Exception as e:
        print(f'❌ FAISS API testing failed: {e}')
        print('Make sure the server is running: uvicorn main:app --reload')
