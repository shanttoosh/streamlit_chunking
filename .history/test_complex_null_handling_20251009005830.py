#!/usr/bin/env python3
"""
Complex Null Handling Testing - 30-60 columns
"""
import pandas as pd
import numpy as np
import random
from backend import suggest_null_strategy_enhanced, apply_null_strategies_enhanced, profile_nulls_enhanced

def generate_complex_dataset(num_rows=1000, num_cols=50):
    """Generate a complex dataset with various data types and null patterns"""
    np.random.seed(42)
    random.seed(42)
    
    data = {}
    
    # Numeric columns (20 columns)
    for i in range(20):
        col_name = f'numeric_{i+1}'
        values = np.random.normal(100, 20, num_rows)
        # Add 10-30% nulls randomly
        null_indices = np.random.choice(num_rows, size=int(num_rows * random.uniform(0.1, 0.3)), replace=False)
        values[null_indices] = np.nan
        data[col_name] = values
    
    # Integer columns (10 columns)
    for i in range(10):
        col_name = f'integer_{i+1}'
        values = np.random.randint(1, 1000, num_rows)
        # Add 5-20% nulls randomly
        null_indices = np.random.choice(num_rows, size=int(num_rows * random.uniform(0.05, 0.2)), replace=False)
        values[null_indices] = np.nan
        data[col_name] = values
    
    # Text/Categorical columns (15 columns)
    categories = ['A', 'B', 'C', 'D', 'E']
    for i in range(15):
        col_name = f'category_{i+1}'
        values = [random.choice(categories) for _ in range(num_rows)]
        # Add 15-40% nulls randomly
        null_indices = np.random.choice(num_rows, size=int(num_rows * random.uniform(0.15, 0.4)), replace=False)
        for idx in null_indices:
            values[idx] = None
        data[col_name] = values
    
    # Boolean columns (5 columns)
    for i in range(5):
        col_name = f'boolean_{i+1}'
        values = [random.choice([True, False]) for _ in range(num_rows)]
        # Add 10-25% nulls randomly
        null_indices = np.random.choice(num_rows, size=int(num_rows * random.uniform(0.1, 0.25)), replace=False)
        for idx in null_indices:
            values[idx] = None
        data[col_name] = values
    
    return pd.DataFrame(data)

def test_complex_dataset():
    """Test complex dataset with 50 columns"""
    print('🧪 TESTING COMPLEX NULL HANDLING - 50 COLUMNS')
    print('=' * 70)
    
    # Generate complex dataset
    df = generate_complex_dataset(num_rows=1000, num_cols=50)
    print(f'Generated dataset: {df.shape[0]} rows × {df.shape[1]} columns')
    
    # Profile nulls
    profile = profile_nulls_enhanced(df)
    print(f'\nNull Profile Summary:')
    print(f'Total columns with nulls: {len(profile[profile["null_count"] > 0])}')
    print(f'Average null percentage: {profile["null_pct"].mean():.1f}%')
    print(f'Max null percentage: {profile["null_pct"].max():.1f}%')
    
    # Show top 10 columns with most nulls
    print(f'\nTop 10 columns with most nulls:')
    top_nulls = profile.nlargest(10, 'null_pct')
    for _, row in top_nulls.iterrows():
        suggestion = suggest_null_strategy_enhanced(row['column'], df[row['column']])
        print(f'  {row["column"]:<15} | {row["dtype"]:<8} | {row["null_count"]:>3} nulls ({row["null_pct"]:>5.1f}%) | Suggested: {suggestion}')
    
    # Generate strategies based on suggestions
    strategies = {}
    for _, row in profile.iterrows():
        if row['null_count'] > 0:
            suggestion = suggest_null_strategy_enhanced(row['column'], df[row['column']])
            if suggestion != 'No change':
                strategies[row['column']] = suggestion
    
    print(f'\nGenerated strategies for {len(strategies)} columns')
    
    # Apply strategies
    print('Applying null handling strategies...')
    result = apply_null_strategies_enhanced(df, strategies)
    
    # Check results
    print(f'\nResults:')
    original_total_nulls = df.isnull().sum().sum()
    result_total_nulls = result.isnull().sum().sum()
    print(f'Total nulls before: {original_total_nulls}')
    print(f'Total nulls after:  {result_total_nulls}')
    print(f'Nulls removed:     {original_total_nulls - result_total_nulls}')
    print(f'Reduction:         {((original_total_nulls - result_total_nulls) / original_total_nulls * 100):.1f}%')
    
    # Show strategy distribution
    strategy_counts = {}
    for strategy in strategies.values():
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print(f'\nStrategy distribution:')
    for strategy, count in sorted(strategy_counts.items()):
        print(f'  {strategy:<10}: {count:>2} columns')
    
    print('\n' + '='*70)

def test_performance():
    """Test performance with large dataset"""
    print('TEST CASE: Performance with Large Dataset')
    print('-' * 50)
    
    import time
    
    # Generate larger dataset
    df = generate_complex_dataset(num_rows=5000, num_cols=60)
    print(f'Testing with {df.shape[0]} rows × {df.shape[1]} columns')
    
    # Profile nulls
    start_time = time.time()
    profile = profile_nulls_enhanced(df)
    profile_time = time.time() - start_time
    print(f'Null profiling time: {profile_time:.3f} seconds')
    
    # Generate strategies
    start_time = time.time()
    strategies = {}
    for _, row in profile.iterrows():
        if row['null_count'] > 0:
            suggestion = suggest_null_strategy_enhanced(row['column'], df[row['column']])
            if suggestion != 'No change':
                strategies[row['column']] = suggestion
    strategy_time = time.time() - start_time
    print(f'Strategy generation time: {strategy_time:.3f} seconds')
    
    # Apply strategies
    start_time = time.time()
    result = apply_null_strategies_enhanced(df, strategies)
    apply_time = time.time() - start_time
    print(f'Strategy application time: {apply_time:.3f} seconds')
    
    total_time = profile_time + strategy_time + apply_time
    print(f'Total processing time: {total_time:.3f} seconds')
    print(f'Processing rate: {df.shape[0] * df.shape[1] / total_time:.0f} cells/second')
    
    print('\n' + '='*70)

if __name__ == "__main__":
    test_complex_dataset()
    test_performance()
    print('✅ Complex test cases completed!')
