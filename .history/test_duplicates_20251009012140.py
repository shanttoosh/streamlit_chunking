#!/usr/bin/env python3
"""
Test Duplicate Row Analysis/Removal System
"""
import pandas as pd
import numpy as np
from backend import analyze_duplicates_enhanced, remove_duplicates_enhanced

def test_duplicate_analysis():
    """Test duplicate analysis functionality"""
    print('🧪 TESTING DUPLICATE ANALYSIS SYSTEM')
    print('=' * 60)
    
    # Test Case 1: Simple duplicates
    print('TEST CASE 1: Simple Duplicates')
    print('-' * 40)
    
    data1 = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'David', 'Alice', 'Eve'],
        'age': [25, 30, 25, 35, 30, 28, 25, 32],
        'city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA', 'Miami', 'NYC', 'Seattle']
    }
    df1 = pd.DataFrame(data1)
    
    print('Original data:')
    print(df1)
    print()
    
    # Analyze duplicates
    analysis1 = analyze_duplicates_enhanced(df1)
    print('Duplicate Analysis:')
    print(f'  Total rows: {analysis1["total_rows"]}')
    print(f'  Duplicate rows: {analysis1["duplicate_rows_count"]}')
    print(f'  Duplicate groups: {analysis1["unique_duplicate_groups"]}')
    print(f'  Duplicate percentage: {analysis1["duplicate_percentage"]:.1f}%')
    print(f'  Has duplicates: {analysis1["has_duplicates"]}')
    print()
    
    # Show duplicate groups
    if analysis1['duplicate_groups']:
        print('Duplicate Groups:')
        for i, group in enumerate(analysis1['duplicate_groups']):
            print(f'  Group {i+1}: {group["count"]} rows')
            print(f'    Values: {group["values"]}')
            print(f'    Indices: {group["indices"]}')
    print()
    
    # Test different removal strategies
    print('Testing Removal Strategies:')
    strategies = ['keep_first', 'keep_last', 'remove_all', 'keep_all']
    
    for strategy in strategies:
        result = remove_duplicates_enhanced(df1, strategy)
        print(f'  {strategy:<12}: {len(df1)} → {len(result)} rows')
    
    print('\n' + '='*60)

def test_complex_duplicates():
    """Test complex duplicate scenarios"""
    print('TEST CASE 2: Complex Duplicate Scenarios')
    print('-' * 40)
    
    # Create complex dataset with various duplicate patterns
    data2 = {
        'user_id': [1, 2, 3, 1, 2, 4, 5, 1, 6, 7],
        'product': ['A', 'B', 'C', 'A', 'B', 'D', 'E', 'A', 'F', 'G'],
        'price': [10.0, 20.0, 30.0, 10.0, 20.0, 40.0, 50.0, 10.0, 60.0, 70.0],
        'category': ['X', 'Y', 'Z', 'X', 'Y', 'W', 'V', 'X', 'U', 'T'],
        'rating': [4.5, 3.8, 4.2, 4.5, 3.8, 4.7, 3.9, 4.5, 4.1, 4.3]
    }
    df2 = pd.DataFrame(data2)
    
    print('Complex dataset:')
    print(df2)
    print()
    
    # Analyze duplicates
    analysis2 = analyze_duplicates_enhanced(df2)
    print('Complex Duplicate Analysis:')
    print(f'  Total rows: {analysis2["total_rows"]}')
    print(f'  Duplicate rows: {analysis2["duplicate_rows_count"]}')
    print(f'  Duplicate groups: {analysis2["unique_duplicate_groups"]}')
    print(f'  Duplicate percentage: {analysis2["duplicate_percentage"]:.1f}%')
    print()
    
    # Test all strategies
    print('All Removal Strategies:')
    for strategy in ['keep_first', 'keep_last', 'remove_all', 'keep_all']:
        result = remove_duplicates_enhanced(df2, strategy)
        removed = len(df2) - len(result)
        print(f'  {strategy:<12}: {len(df2)} → {len(result)} rows (removed {removed})')
        
        if len(result) > 0:
            print(f'    Sample result: {result.iloc[0].to_dict()}')
    print()
    
    print('='*60)

def test_no_duplicates():
    """Test dataset with no duplicates"""
    print('TEST CASE 3: No Duplicates')
    print('-' * 40)
    
    data3 = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['NYC', 'LA', 'Chicago', 'Miami', 'Seattle']
    }
    df3 = pd.DataFrame(data3)
    
    print('No duplicates dataset:')
    print(df3)
    print()
    
    # Analyze duplicates
    analysis3 = analyze_duplicates_enhanced(df3)
    print('No Duplicates Analysis:')
    print(f'  Total rows: {analysis3["total_rows"]}')
    print(f'  Duplicate rows: {analysis3["duplicate_rows_count"]}')
    print(f'  Duplicate groups: {analysis3["unique_duplicate_groups"]}')
    print(f'  Has duplicates: {analysis3["has_duplicates"]}')
    print()
    
    # Test removal (should return unchanged)
    result = remove_duplicates_enhanced(df3, 'keep_first')
    print(f'Removal test: {len(df3)} → {len(result)} rows (unchanged: {len(df3) == len(result)})')
    
    print('\n' + '='*60)

def test_large_dataset():
    """Test with larger dataset"""
    print('TEST CASE 4: Large Dataset Performance')
    print('-' * 40)
    
    import time
    
    # Create larger dataset with duplicates
    np.random.seed(42)
    n_rows = 1000
    n_duplicates = 200
    
    # Generate unique data
    data4 = {
        'id': range(n_rows),
        'value1': np.random.randn(n_rows),
        'value2': np.random.randint(1, 100, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows)
    }
    df4 = pd.DataFrame(data4)
    
    # Add duplicates by copying some rows
    duplicate_indices = np.random.choice(n_rows, n_duplicates, replace=True)
    duplicate_rows = df4.iloc[duplicate_indices].copy()
    duplicate_rows.index = range(n_rows, n_rows + n_duplicates)
    df4_with_duplicates = pd.concat([df4, duplicate_rows], ignore_index=True)
    
    print(f'Large dataset: {len(df4_with_duplicates)} rows (added {n_duplicates} duplicates)')
    
    # Test performance
    start_time = time.time()
    analysis4 = analyze_duplicates_enhanced(df4_with_duplicates)
    analysis_time = time.time() - start_time
    
    start_time = time.time()
    result4 = remove_duplicates_enhanced(df4_with_duplicates, 'keep_first')
    removal_time = time.time() - start_time
    
    print(f'Analysis time: {analysis_time:.3f} seconds')
    print(f'Removal time: {removal_time:.3f} seconds')
    print(f'Total time: {analysis_time + removal_time:.3f} seconds')
    print()
    
    print('Results:')
    print(f'  Original rows: {len(df4_with_duplicates)}')
    print(f'  After removal: {len(result4)}')
    print(f'  Duplicates found: {analysis4["duplicate_rows_count"]}')
    print(f'  Duplicate groups: {analysis4["unique_duplicate_groups"]}')
    
    print('\n' + '='*60)

if __name__ == "__main__":
    test_duplicate_analysis()
    test_complex_duplicates()
    test_no_duplicates()
    test_large_dataset()
    print('✅ All duplicate handling tests completed!')
