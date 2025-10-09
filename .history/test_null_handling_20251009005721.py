#!/usr/bin/env python3
"""
Comprehensive Null Handling Testing
"""
import pandas as pd
import numpy as np
from backend import suggest_null_strategy_enhanced, apply_null_strategies_enhanced, profile_nulls_enhanced

def test_simple_cases():
    """Test simple null handling cases"""
    print('🧪 TESTING SIMPLE NULL HANDLING CASES')
    print('=' * 60)
    
    # Test Case 1: Basic numeric columns
    print('TEST CASE 1: Basic Numeric Columns')
    print('-' * 40)
    data1 = {
        'temperature': [25.5, None, 30.2, None, 28.9, 27.1, None],
        'pressure': [1013.25, None, 1015.30, None, 1012.80, 1014.50, None],
        'humidity': [60.5, 65.2, None, 58.9, 62.1, None, 59.8]
    }
    df1 = pd.DataFrame(data1)
    
    # Profile nulls
    profile1 = profile_nulls_enhanced(df1)
    print('Null Profile:')
    for _, row in profile1.iterrows():
        suggestion = suggest_null_strategy_enhanced(row['column'], df1[row['column']])
        print(f'  {row["column"]:<12} | {row["dtype"]:<8} | {row["null_count"]:>2} nulls ({row["null_pct"]:>5.1f}%) | Suggested: {suggestion}')
    
    # Test strategies
    strategies1 = {
        'temperature': 'median',
        'pressure': 'mean',
        'humidity': 'mode'
    }
    
    result1 = apply_null_strategies_enhanced(df1, strategies1)
    print('\nAfter applying strategies:')
    for col in df1.columns:
        original_nulls = df1[col].isnull().sum()
        result_nulls = result1[col].isnull().sum()
        print(f'  {col:<12} | {original_nulls} → {result_nulls} nulls | Strategy: {strategies1[col]}')
    
    print('\n' + '='*60)

def test_text_cases():
    """Test text/categorical null handling"""
    print('TEST CASE 2: Text/Categorical Columns')
    print('-' * 40)
    
    data2 = {
        'machine_id': ['M001', None, 'M002', None, 'M003', 'M004', None],
        'status': ['active', 'inactive', None, 'active', None, 'inactive', 'active'],
        'category': ['A', 'B', None, 'A', 'C', None, 'B'],
        'notes': ['Good', None, 'Excellent', None, 'Average', 'Poor', None]
    }
    df2 = pd.DataFrame(data2)
    
    # Profile nulls
    profile2 = profile_nulls_enhanced(df2)
    print('Null Profile:')
    for _, row in profile2.iterrows():
        suggestion = suggest_null_strategy_enhanced(row['column'], df2[row['column']])
        print(f'  {row["column"]:<12} | {row["dtype"]:<8} | {row["null_count"]:>2} nulls ({row["null_pct"]:>5.1f}%) | Suggested: {suggestion}')
    
    # Test strategies
    strategies2 = {
        'machine_id': 'mode',
        'status': 'mode',
        'category': 'unknown',
        'notes': 'unknown'
    }
    
    result2 = apply_null_strategies_enhanced(df2, strategies2)
    print('\nAfter applying strategies:')
    for col in df2.columns:
        original_nulls = df2[col].isnull().sum()
        result_nulls = result2[col].isnull().sum()
        print(f'  {col:<12} | {original_nulls} → {result_nulls} nulls | Strategy: {strategies2[col]}')
    
    print('\n' + '='*60)

def test_mixed_cases():
    """Test mixed data types"""
    print('TEST CASE 3: Mixed Data Types')
    print('-' * 40)
    
    data3 = {
        'id': [1, 2, 3, 4, 5, 6, 7],
        'score': [85.5, None, 92.3, None, 78.1, 88.7, None],
        'count': [10, None, 15, 20, None, 25, 30],
        'flag': [True, None, False, True, None, False, True],
        'text': ['Good', None, 'Excellent', None, 'Average', 'Poor', None]
    }
    df3 = pd.DataFrame(data3)
    
    # Profile nulls
    profile3 = profile_nulls_enhanced(df3)
    print('Null Profile:')
    for _, row in profile3.iterrows():
        suggestion = suggest_null_strategy_enhanced(row['column'], df3[row['column']])
        print(f'  {row["column"]:<12} | {row["dtype"]:<8} | {row["null_count"]:>2} nulls ({row["null_pct"]:>5.1f}%) | Suggested: {suggestion}')
    
    # Test strategies
    strategies3 = {
        'score': 'median',
        'count': 'zero',
        'flag': 'mode',
        'text': 'unknown'
    }
    
    result3 = apply_null_strategies_enhanced(df3, strategies3)
    print('\nAfter applying strategies:')
    for col in df3.columns:
        if col in strategies3:
            original_nulls = df3[col].isnull().sum()
            result_nulls = result3[col].isnull().sum()
            print(f'  {col:<12} | {original_nulls} → {result_nulls} nulls | Strategy: {strategies3[col]}')
    
    print('\n' + '='*60)

if __name__ == "__main__":
    test_simple_cases()
    test_text_cases()
    test_mixed_cases()
    print('✅ Simple test cases completed!')
