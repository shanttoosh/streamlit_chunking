#!/usr/bin/env python3
"""
Test script to verify if the preview API endpoint is working correctly
"""

import requests
import pandas as pd
import io

# Test CSV preview API
def test_csv_preview():
    print("=== Testing CSV Preview API ===")
    
    # Create a test CSV
    test_data = {
        'name': ['Alice', 'Bob', None, 'Charlie'],
        'age': [25, None, 35, 40],
        'city': ['New York', 'London', 'Paris', None],
        'score': [85.5, 92.0, 78.5, 88.0]
    }
    df = pd.DataFrame(test_data)
    
    # Save to in-memory CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue().encode('utf-8')
    
    try:
        response = requests.post(
            "http://localhost:8000/preview/data",
            files={"file": ("test.csv", csv_content, "text/csv")},
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print("✅ CSV Preview API Success!")
            print(f"Response keys: {list(result.keys())}")
            
            if "dtype_preview" in result:
                print(f"Dtype preview: {result['dtype_preview']}")
            if "null_preview" in result:
                print(f"Null preview: {result['null_preview']}")
            print(f"Total rows: {result.get('total_rows')}")
            print(f"Total columns: {result.get('total_columns')}")
        else:
            print(f"❌ Error: {result}")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

def test_api_health():
    """Test if API is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: API is running")
            return True
        else:
            print("ERROR: API health check failed")
            return False
    except Exception as e:
        print(f"ERROR: API not accessible: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Preview API Endpoint...")
    
    if test_api_health():
        test_csv_preview()
    else:
        print("Please start the API server first with: uvicorn main:app --reload")
