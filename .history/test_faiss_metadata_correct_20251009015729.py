#!/usr/bin/env python3
"""
Test FAISS Metadata with Correct Case
"""
import requests
import json
import pandas as pd

def test_faiss_metadata_correct_case():
    """Test FAISS metadata filtering with correct case"""
    print('🧪 TESTING FAISS METADATA WITH CORRECT CASE')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Test queries with correct case (lowercase)
    test_queries = [
        {
            'name': 'Filter by city (nyc)',
            'query': 'system performance',
            'k': 5,
            'metadata_filter': '{"city_mode": "nyc"}'
        },
        {
            'name': 'Filter by city (la)',
            'query': 'data analysis',
            'k': 3,
            'metadata_filter': '{"city_mode": "la"}'
        },
        {
            'name': 'Filter by city (chicago)',
            'query': 'processing information',
            'k': 3,
            'metadata_filter': '{"city_mode": "chicago"}'
        },
        {
            'name': 'Filter by score mean (exact match)',
            'query': 'results improvement',
            'k': 3,
            'metadata_filter': '{"score_mean": 91.75}'
        },
        {
            'name': 'Multiple filters (nyc + specific score)',
            'query': 'system efficiency',
            'k': 3,
            'metadata_filter': '{"city_mode": "nyc", "score_mean": 81.8}'
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
    print('Testing FAISS metadata with correct case...')
    print('Make sure the FastAPI server is running on localhost:8000')
    print()
    
    try:
        test_faiss_metadata_correct_case()
        print('✅ FAISS metadata testing with correct case completed!')
    except Exception as e:
        print(f'❌ FAISS metadata testing failed: {e}')
