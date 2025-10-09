#!/usr/bin/env python3
"""
Test UI Metadata Storage Integration
"""
import requests
import json
import pandas as pd

def test_ui_metadata_storage_integration():
    """Test the UI metadata storage integration"""
    print('🧪 TESTING UI METADATA STORAGE INTEGRATION')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Test data
    test_data = {
        'text': [
            'The system is running smoothly in New York',
            'We are analyzing data carefully in Los Angeles',
            'The machines are working efficiently in Chicago'
        ],
        'score': [85.5, 92.3, 78.1],
        'city': ['NYC', 'LA', 'Chicago'],
        'category': ['A', 'B', 'A']
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
    
    # Step 2: Test ChromaDB chunking with metadata
    print('STEP 2: ChromaDB Chunking with Metadata')
    print('-' * 30)
    try:
        chunk_data = {
            'chunk_method': 'document',
            'key_column': 'city',
            'token_limit': 1000,
            'store_metadata': 'true',
            'numeric_columns': 1,  # Include score column
            'categorical_columns': 1  # Include category column
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
    
    # Step 4: Test both storage types
    storage_types = [
        {"name": "ChromaDB", "type": "chroma"},
        {"name": "FAISS", "type": "faiss"}
    ]
    
    for storage in storage_types:
        print(f'STEP 4: Store in {storage["name"]}')
        print('-' * 30)
        try:
            store_data = {'storage_type': storage["type"]}
            response = requests.post(f"{base_url}/deep_config/store", data=store_data)
            print(f'Storage status: {response.status_code}')
            if response.status_code == 200:
                result = response.json()
                print(f'Storage result: {result.get("status", "unknown")}')
                print(f'Storage type: {result.get("storage_type", "unknown")}')
                print(f'Total vectors: {result.get("total_vectors", 0)}')
            else:
                print(f'Storage error: {response.text}')
                continue
        except Exception as e:
            print(f'Storage error: {e}')
            continue
        
        print()
        
        # Step 5: Test metadata filtering
        print(f'STEP 5: Test {storage["name"]} Metadata Filtering')
        print('-' * 30)
        
        test_queries = [
            {
                'name': 'Filter by city (nyc)',
                'query': 'system performance',
                'k': 3,
                'metadata_filter': '{"city_mode": "nyc"}'
            },
            {
                'name': 'Filter by category (A)',
                'query': 'data analysis',
                'k': 3,
                'metadata_filter': '{"category_mode": "A"}'
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
                    print(f'  Store type: {result.get("store_type", "unknown")}')
                else:
                    print(f'  Error: {response.text}')
                    
            except Exception as e:
                print(f'  Error: {e}')
        
        print('=' * 60)
        print()

if __name__ == "__main__":
    print('Testing UI metadata storage integration...')
    print('Make sure the FastAPI server is running on localhost:8000')
    print()
    
    try:
        test_ui_metadata_storage_integration()
        print('✅ UI metadata storage integration testing completed!')
    except Exception as e:
        print(f'❌ UI metadata storage integration testing failed: {e}')
        print('Make sure the server is running: uvicorn main:app --reload')
