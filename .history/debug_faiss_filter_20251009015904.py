#!/usr/bin/env python3
"""
Debug FAISS Filtering
"""
import requests
import json

def debug_faiss_filter():
    """Debug FAISS filtering"""
    print('🔍 DEBUGGING FAISS FILTERING')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Test with a simple filter
    query_data = {
        'query': 'system performance',
        'k': 5,
        'metadata_filter': '{"city_mode": "nyc"}'
    }
    
    print(f'Query data: {query_data}')
    
    try:
        response = requests.post(f"{base_url}/retrieve_with_metadata", data=query_data)
        print(f'Response status: {response.status_code}')
        print(f'Response text: {response.text}')
        
        if response.status_code == 200:
            result = response.json()
            print(f'Result: {json.dumps(result, indent=2)}')
        else:
            print(f'Error response: {response.text}')
            
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    debug_faiss_filter()
