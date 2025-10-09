#!/usr/bin/env python3
"""
Test Text Normalization System (Stemming, Lemmatization, Skip)
"""
import pandas as pd
import numpy as np
import time
from backend import process_text_enhanced

def test_small_dataset():
    """Test text normalization with small dataset"""
    print('🧪 TESTING TEXT NORMALIZATION - SMALL DATASET')
    print('=' * 60)
    
    # Create small test dataset
    data = {
        'id': [1, 2, 3, 4, 5],
        'description': [
            'The cats are running quickly through the garden',
            'I am better at programming than yesterday',
            'The machines are working efficiently and smoothly',
            'We need to analyze the data carefully and thoroughly',
            'The system is processing information automatically'
        ],
        'notes': [
            'Running tests and checking results',
            'Better performance and improved accuracy',
            'Working machines and efficient processes',
            'Analyzing data and processing information',
            'Automatic processing and system optimization'
        ],
        'category': ['A', 'B', 'C', 'D', 'E'],  # Non-text column
        'score': [85.5, 92.3, 78.1, 88.7, 91.2]  # Numeric column
    }
    
    df = pd.DataFrame(data)
    print('Original data:')
    print(df[['description', 'notes']].head())
    print()
    
    # Test all methods
    methods = ['none', 'lemmatize', 'stem']
    
    for method in methods:
        print(f'TESTING METHOD: {method.upper()}')
        print('-' * 40)
        
        start_time = time.time()
        try:
            result = process_text_enhanced(df.copy(), method)
            processing_time = time.time() - start_time
            
            print(f'Processing time: {processing_time:.3f} seconds')
            print('Results:')
            print(result[['description', 'notes']].head())
            
            # Show before/after comparison
            print('BEFORE vs AFTER comparison:')
            for i in range(3):
                print(f'Row {i+1}:')
                print(f'  BEFORE: {df.iloc[i]["description"]}')
                print(f'  AFTER:  {result.iloc[i]["description"]}')
                print()
                
        except Exception as e:
            print(f'Error: {e}')
            print('This is expected if spaCy/NLTK is not installed')
        
        print('=' * 60)
        print()

def test_medium_dataset():
    """Test text normalization with medium dataset"""
    print('🧪 TESTING TEXT NORMALIZATION - MEDIUM DATASET')
    print('=' * 60)
    
    # Create medium test dataset
    np.random.seed(42)
    n_rows = 100
    
    # Generate text data with various patterns
    base_texts = [
        'The system is running smoothly and efficiently',
        'We are analyzing the data carefully and thoroughly',
        'The machines are working better than expected',
        'Processing information automatically and accurately',
        'The results show significant improvement over time',
        'We need to optimize the performance and reliability',
        'The analysis reveals important patterns and trends',
        'Implementing changes to enhance functionality',
        'The system processes data quickly and effectively',
        'Monitoring performance and adjusting parameters'
    ]
    
    # Create variations
    variations = [
        'running', 'ran', 'runs', 'run',
        'analyzing', 'analyzed', 'analyzes', 'analyze',
        'working', 'worked', 'works', 'work',
        'processing', 'processed', 'processes', 'process',
        'improving', 'improved', 'improves', 'improve',
        'optimizing', 'optimized', 'optimizes', 'optimize',
        'revealing', 'revealed', 'reveals', 'reveal',
        'implementing', 'implemented', 'implements', 'implement',
        'monitoring', 'monitored', 'monitors', 'monitor'
    ]
    
    descriptions = []
    notes = []
    
    for i in range(n_rows):
        base_text = np.random.choice(base_texts)
        variation = np.random.choice(variations)
        descriptions.append(f'{base_text} with {variation} capabilities')
        notes.append(f'Testing {variation} functionality and performance')
    
    data = {
        'id': range(1, n_rows + 1),
        'description': descriptions,
        'notes': notes,
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'score': np.random.uniform(70, 100, n_rows)
    }
    
    df = pd.DataFrame(data)
    print(f'Medium dataset: {len(df)} rows')
    print('Sample data:')
    print(df[['description', 'notes']].head(3))
    print()
    
    # Test performance
    methods = ['none', 'lemmatize', 'stem']
    
    for method in methods:
        print(f'TESTING METHOD: {method.upper()}')
        print('-' * 40)
        
        start_time = time.time()
        try:
            result = process_text_enhanced(df.copy(), method)
            processing_time = time.time() - start_time
            
            print(f'Processing time: {processing_time:.3f} seconds')
            print(f'Processing rate: {len(df) / processing_time:.0f} rows/second')
            
            # Show sample results
            print('Sample results:')
            for i in range(2):
                print(f'Row {i+1}:')
                print(f'  BEFORE: {df.iloc[i]["description"]}')
                print(f'  AFTER:  {result.iloc[i]["description"]}')
                print()
                
        except Exception as e:
            print(f'Error: {e}')
        
        print('=' * 60)
        print()

def test_large_dataset():
    """Test text normalization with large dataset"""
    print('🧪 TESTING TEXT NORMALIZATION - LARGE DATASET')
    print('=' * 60)
    
    # Create large test dataset
    np.random.seed(42)
    n_rows = 1000
    
    # Generate text data
    base_texts = [
        'The system is running smoothly and efficiently',
        'We are analyzing the data carefully and thoroughly',
        'The machines are working better than expected',
        'Processing information automatically and accurately',
        'The results show significant improvement over time',
        'We need to optimize the performance and reliability',
        'The analysis reveals important patterns and trends',
        'Implementing changes to enhance functionality',
        'The system processes data quickly and effectively',
        'Monitoring performance and adjusting parameters'
    ]
    
    descriptions = []
    notes = []
    
    for i in range(n_rows):
        base_text = np.random.choice(base_texts)
        descriptions.append(f'{base_text} for row {i+1}')
        notes.append(f'Processing data for analysis and optimization')
    
    data = {
        'id': range(1, n_rows + 1),
        'description': descriptions,
        'notes': notes,
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
        'score': np.random.uniform(70, 100, n_rows)
    }
    
    df = pd.DataFrame(data)
    print(f'Large dataset: {len(df)} rows')
    print('Sample data:')
    print(df[['description', 'notes']].head(2))
    print()
    
    # Test performance
    methods = ['none', 'lemmatize', 'stem']
    
    for method in methods:
        print(f'TESTING METHOD: {method.upper()}')
        print('-' * 40)
        
        start_time = time.time()
        try:
            result = process_text_enhanced(df.copy(), method)
            processing_time = time.time() - start_time
            
            print(f'Processing time: {processing_time:.3f} seconds')
            print(f'Processing rate: {len(df) / processing_time:.0f} rows/second')
            print(f'Text columns processed: 2')
            print(f'Total text processing: {len(df) * 2} cells')
            print(f'Cells/second: {(len(df) * 2) / processing_time:.0f}')
            
            # Show sample results
            print('Sample results:')
            for i in range(2):
                print(f'Row {i+1}:')
                print(f'  BEFORE: {df.iloc[i]["description"]}')
                print(f'  AFTER:  {result.iloc[i]["description"]}')
                print()
                
        except Exception as e:
            print(f'Error: {e}')
        
        print('=' * 60)
        print()

def test_edge_cases():
    """Test edge cases and error handling"""
    print('🧪 TESTING EDGE CASES')
    print('=' * 60)
    
    # Test with no text columns
    data1 = {
        'id': [1, 2, 3],
        'score': [85.5, 92.3, 78.1],
        'count': [10, 20, 30]
    }
    df1 = pd.DataFrame(data1)
    
    print('Test 1: No text columns')
    print('-' * 30)
    try:
        result1 = process_text_enhanced(df1.copy(), 'lemmatize')
        print('Result: No changes (no text columns)')
        print(f'Data unchanged: {df1.equals(result1)}')
    except Exception as e:
        print(f'Error: {e}')
    print()
    
    # Test with empty text
    data2 = {
        'id': [1, 2, 3],
        'text': ['', '   ', None],
        'description': ['Some text', '', 'More text']
    }
    df2 = pd.DataFrame(data2)
    
    print('Test 2: Empty text handling')
    print('-' * 30)
    try:
        result2 = process_text_enhanced(df2.copy(), 'lemmatize')
        print('Results:')
        print(result2[['text', 'description']])
    except Exception as e:
        print(f'Error: {e}')
    print()
    
    # Test with mixed data types
    data3 = {
        'id': [1, 2, 3],
        'text': ['Running tests', 123, 'Analyzing data'],
        'description': ['Better performance', 'Good results', 456]
    }
    df3 = pd.DataFrame(data3)
    
    print('Test 3: Mixed data types')
    print('-' * 30)
    try:
        result3 = process_text_enhanced(df3.copy(), 'lemmatize')
        print('Results:')
        print(result3[['text', 'description']])
    except Exception as e:
        print(f'Error: {e}')
    
    print('=' * 60)

if __name__ == "__main__":
    test_small_dataset()
    test_medium_dataset()
    test_large_dataset()
    test_edge_cases()
    print('✅ All text normalization tests completed!')
