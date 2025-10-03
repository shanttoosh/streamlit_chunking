#!/usr/bin/env python3
"""
Test the preview data structure and display logic
"""

# Simulate the preview data structure we get from the API
import pandas as pd

def test_preview_display_logic():
    """Test exactly what preview data we get and how it's processed"""
    
    # This is the actual structure from our API test
    preview_data = {
        "source_type": "csv",
        "source_info": {"filename": "test.csv"},
        "dtype_preview": {
            "columns": ["column_name", "current_dtype", "sample_values"],
            "data": [
                {"column_name": "name", "current_dtype": "object", "sample_values": ["alice", "bob", ""]},
                {"column_name": "age", "current_dtype": "float64", "sample_values": [25.0, 35.0, 40.0]},
                {"column_name": "city", "current_dtype": "object", "sample_values": ["new york", "london", "paris"]}
            ],
            "total_rows": 3
        },
        "null_preview": {
            "columns": ["column_name", "null_count", "null_percentage"],
            "data": [
                {"column_name": "age", "null_count": 1, "null_percentage": 25.0},
                {"column_name": "name", "null_count": 0, "null_percentage": 0.0},
                {"column_name": "city", "null_count": 0, "null_percentage": 0.0}
            ],
            "total_rows": 3
        },
        "total_rows": 4,
        "total_columns": 3
    }
    
    print("=== Testing Preview Data Structure ===")
    print(f"Preview data keys: {list(preview_data.keys())}")
    
    # Test step 1: Check if preview_data is valid
    if preview_data and "error" not in preview_data:
        print("✓ Preview data is valid")
    else:
        print("✗ Preview data is invalid")
        return
    
    # Test step 2: Check dtype_preview
    if "dtype_preview" in preview_data and preview_data["dtype_preview"]:
        print("✓ dtype_preview exists and is not None")
        dtype_data = preview_data["dtype_preview"]["data"]
        print(f"  dtype_data length: {len(dtype_data)}")
        print(f"  dtype_data sample: {dtype_data[0] if dtype_data else 'No data'}")
        
        # Test DataFrame creation
        if dtype_data:
            dtype_df = pd.DataFrame(dtype_data)
            print(f"  DataFrame created: {not dtype_df.empty}")
            print(f"  DataFrame shape: {dtype_df.shape}")
            print(f"  DataFrame columns: {list(dtype_df.columns)}")
        else:
            print("  No dtype_data available")
    else:
        print("✗ dtype_preview missing or None")
    
    # Test step 3: Check null_preview
    if "null_preview" in preview_data and preview_data["null_preview"]:
        print("✓ null_preview exists and is not None")
        null_data = preview_data["null_preview"]["data"]
        print(f"  null_data length: {len(null_data)}")
        print(f"  null_data sample: {null_data[0] if null_data else 'No data'}")
        
        # Test DataFrame creation
        if null_data:
            null_df = pd.DataFrame(null_data)
            print(f"  DataFrame created: {not null_df.empty}")
            print(f"  DataFrame shape: {null_df.shape}")
            print(f"  DataFrame columns: {list(null_df.columns)}")
        else:
            print("  No null_data available")
    else:
        print("✗ null_preview missing or None")
    
    # Test step 4: Column extraction for UI
    print("\n=== Testing Column Extraction ===")
    
    # Config-1 null handling columns
    if "null_preview" in preview_data and preview_data["null_preview"]:
        columns = []
        for item in preview_data["null_preview"]["data"]:
            if isinstance(item, dict) and "column_name" in item:
                columns.append(item["column_name"])
        
        print(f"✓ null_preview columns extracted: {columns}")
        
        if not columns:
            print("⚠ No columns found - this might be the issue!")
    else:
        print("✗ No null_preview to extract columns from")
    
    # Deep Mode dtype columns
    if "dtype_preview" in preview_data and preview_data["dtype_preview"]:
        columns = []
        for item in preview_data["dtype_preview"]["data"]:
            if isinstance(item, dict) and "column_name" in item:
                columns.append(item["column_name"])
        
        print(f"✓ dtype_conversion columns extracted: {columns}")
        
        if not columns:
            print("⚠ No columns found - this might be the issue!")
    else:
        print("✗ No dtype_preview to extract columns from")

if __name__ == "__main__":
    test_preview_display_logic()
