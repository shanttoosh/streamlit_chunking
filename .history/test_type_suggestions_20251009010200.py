#!/usr/bin/env python3
"""
Comprehensive Type Suggestion Testing
"""
import pandas as pd
import numpy as np
from backend import suggest_null_strategy_enhanced

def get_smart_suggestion(col_name, col_data):
    """Copy of the frontend function for testing"""
    col_name_lower = col_name.lower()
    
    # Handle datetime columns by name
    if any(word in col_name_lower for word in ['date', 'time', 'created', 'updated', 'timestamp', 'birth', 'join']):
        return 'datetime'
    
    # Handle boolean columns by name
    if any(word in col_name_lower for word in ['flag', 'is_', 'has_', 'active', 'enabled', 'status', 'complaint']):
        return 'boolean'
    
    # Handle specific patterns by name
    if any(word in col_name_lower for word in ['phone', 'email', 'ip', 'zip', 'address']):
        return 'object'
    
    # Analyze data patterns for object columns
    if col_data.dtype == 'object':
        sample_values = col_data.dropna().head(30)
        if len(sample_values) == 0:
            return 'object'
        
        # Check for datetime patterns
        date_patterns = []
        for val in sample_values:
            val_str = str(val).strip()
            if (any(char in val_str for char in ['-', '/', ':']) and 
                len(val_str) > 8 and 
                (val_str.count('-') >= 2 or val_str.count('/') >= 2 or ':' in val_str or 'T' in val_str)):
                date_patterns.append(val_str)
        
        if len(date_patterns) > len(sample_values) * 0.5:
            return 'datetime'
        
        # Check for boolean patterns
        bool_values = []
        for val in sample_values:
            val_str = str(val).lower().strip()
            if val_str in ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n', 't', 'f']:
                bool_values.append(val_str)
        
        if len(bool_values) > len(sample_values) * 0.6:
            return 'boolean'
        
        # Check for numeric patterns
        numeric_values = []
        for val in sample_values:
            try:
                clean_val = str(val).replace(',', '').replace('$', '').replace('%', '').replace('USD', '').replace('€', '').strip()
                float(clean_val)
                numeric_values.append(val)
            except:
                pass
        
        if (len(numeric_values) > len(sample_values) * 0.7 and 
            not any(word in col_name_lower for word in ['id', 'key', 'code', 'ref', 'phone', 'email', 'ip', 'zip'])):
            return 'float64'
        
        unique_ratio = col_data.nunique() / len(col_data)
        if unique_ratio < 0.2 and col_data.nunique() < 20:
            return 'object'
        
        return 'object'
    
    # Handle numeric columns
    elif str(col_data.dtype) == 'int64':
        unique_ratio = col_data.nunique() / len(col_data)
        if (unique_ratio > 0.9 and 
            any(word in col_name_lower for word in ['id', 'key', 'code', 'ref', 'number'])):
            return 'object'
        else:
            return 'int64'
    
    elif str(col_data.dtype) == 'float64':
        return 'float64'
    
    return 'object'

def test_type_suggestions():
    """Test type suggestions with various data patterns"""
    print('🧪 TESTING TYPE SUGGESTIONS')
    print('=' * 60)
    
    # Test Case 1: Clear data types
    print('TEST CASE 1: Clear Data Types')
    print('-' * 40)
    
    data1 = {
        'user_id': [1001, 1002, 1003, 1004, 1005],  # High cardinality ID
        'age': [25, 30, 35, 28, 32],  # Low cardinality numeric
        'salary': [50000.0, 60000.0, 55000.0, 65000.0, 58000.0],  # Float
        'is_active': [True, False, True, True, False],  # Boolean
        'created_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],  # Date
        'category': ['A', 'B', 'A', 'C', 'B'],  # Categorical
        'description': ['Good product', 'Excellent quality', 'Average item', 'Great value', 'Nice design']  # Text
    }
    df1 = pd.DataFrame(data1)
    
    for col in df1.columns:
        current_type = str(df1[col].dtype)
        suggested_type = get_smart_suggestion(col, df1[col])
        print(f'  {col:<15} | {current_type:<10} → {suggested_type}')
    
    print('\n' + '='*60)
    
    # Test Case 2: Ambiguous patterns
    print('TEST CASE 2: Ambiguous Patterns')
    print('-' * 40)
    
    data2 = {
        'mixed_numeric': ['123.45', '456.78', '789.01', '999.99', '111.11'],  # Numeric strings
        'mixed_boolean': ['true', 'false', 'yes', 'no', '1'],  # Boolean strings
        'mixed_dates': ['2024-01-15', '01/15/2024', '15-Jan-2024', '2024/01/15', 'Jan 15, 2024'],  # Date strings
        'currency': ['$1,234.56', 'USD 1,234.56', '1,234.56 USD', '€1,234.56', '$999.99'],  # Currency
        'percentage': ['85.5%', '92.3%', '78.1%', '88.7%', '91.2%'],  # Percentage
        'scientific': ['1.23e-4', '2.45E+6', '3.67e2', '4.89E-1', '5.12e0'],  # Scientific notation
        'phone': ['+1-555-123-4567', '(555) 123-4567', '555.123.4567', '5551234567', '+44-20-7946-0958'],  # Phone
        'email': ['user@example.com', 'test@domain.org', 'admin@site.net', 'john@company.com', 'jane@university.edu'],  # Email
        'ip': ['192.168.1.1', '10.0.0.1', '172.16.0.1', '8.8.8.8', '1.1.1.1'],  # IP addresses
        'zip': ['12345', '12345-6789', 'K1A 0A6', 'SW1A 1AA', '90210']  # Zip codes
    }
    df2 = pd.DataFrame(data2)
    
    for col in df2.columns:
        current_type = str(df2[col].dtype)
        suggested_type = get_smart_suggestion(col, df2[col])
        print(f'  {col:<15} | {current_type:<10} → {suggested_type}')
    
    print('\n' + '='*60)
    
    # Test Case 3: Edge cases
    print('TEST CASE 3: Edge Cases')
    print('-' * 40)
    
    data3 = {
        'high_cardinality_id': [f'ID_{i:06d}' for i in range(5)],  # High cardinality string IDs
        'low_cardinality_numeric': [1, 2, 3, 1, 2],  # Low cardinality numeric
        'mostly_numeric': ['123.45', '456.78', '789.01', 'invalid', '999.99'],  # Mostly numeric with invalid
        'mostly_boolean': ['true', 'false', 'yes', 'no', 'maybe'],  # Mostly boolean with invalid
        'mostly_dates': ['2024-01-15', '2024-01-16', '2024-01-17', 'invalid', '2024-01-18'],  # Mostly dates with invalid
        'empty_column': [None, None, None, None, None],  # All nulls
        'single_value': ['A', 'A', 'A', 'A', 'A'],  # Single value
        'mixed_case': ['High', 'LOW', 'Medium', 'HIGH', 'low'],  # Mixed case
        'with_nulls': ['A', None, 'B', None, 'C'],  # With nulls
        'numeric_id': [1000001, 1000002, 1000003, 1000004, 1000005]  # Numeric ID
    }
    df3 = pd.DataFrame(data3)
    
    for col in df3.columns:
        current_type = str(df3[col].dtype)
        suggested_type = get_smart_suggestion(col, df3[col])
        null_count = df3[col].isnull().sum()
        unique_count = df3[col].nunique()
        print(f'  {col:<20} | {current_type:<10} → {suggested_type:<10} | nulls: {null_count}, unique: {unique_count}')
    
    print('\n' + '='*60)

def test_null_suggestions():
    """Test null handling suggestions"""
    print('TEST CASE 4: Null Handling Suggestions')
    print('-' * 40)
    
    data4 = {
        'temperature': [25.5, None, 30.2, None, 28.9, 27.1, None],  # 42.9% nulls
        'pressure': [1013.25, None, 1015.30, None, 1012.80, 1014.50, None],  # 42.9% nulls
        'machine_id': ['M001', None, 'M002', None, 'M003', 'M004', None],  # 42.9% nulls
        'status': ['active', 'inactive', None, 'active', None, 'inactive', 'active'],  # 28.6% nulls
        'count': [10, None, 15, 20, None, 25, 30],  # 28.6% nulls
        'flag': [True, None, False, True, None, False, True],  # 28.6% nulls
        'created_date': ['2024-01-15', None, '2024-01-16', '2024-01-17', None, '2024-01-18', '2024-01-19'],  # 28.6% nulls
        'notes': ['Good', None, 'Excellent', None, 'Average', 'Poor', None]  # 42.9% nulls
    }
    df4 = pd.DataFrame(data4)
    
    for col in df4.columns:
        null_count = df4[col].isnull().sum()
        null_pct = (null_count / len(df4)) * 100
        suggestion = suggest_null_strategy_enhanced(col, df4[col])
        print(f'  {col:<15} | {null_count:>2} nulls ({null_pct:>5.1f}%) | Suggested: {suggestion}')
    
    print('\n' + '='*60)

if __name__ == "__main__":
    test_type_suggestions()
    test_null_suggestions()
    print('✅ Type suggestion tests completed!')
