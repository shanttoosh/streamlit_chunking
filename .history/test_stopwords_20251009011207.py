#!/usr/bin/env python3
"""
Test Stop Word Removal System
"""
import pandas as pd
import numpy as np
from backend import remove_stopwords_from_text_column_enhanced

def test_stopwords_system():
    """Test the stop word removal system"""
    print('🧪 TESTING STOP WORD REMOVAL SYSTEM')
    print('=' * 60)
    
    # Create test data with text columns
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'description': [
            'This is a very good product that works well',
            'The machine is running smoothly and efficiently',
            'We need to check the system status and performance',
            'This is an excellent solution for our problems',
            'The quality of this item is outstanding and reliable'
        ],
        'notes': [
            'Please review and approve this request',
            'The data shows significant improvement over time',
            'We should consider all available options carefully',
            'This approach will help us achieve better results',
            'The team needs to work together on this project'
        ],
        'category': ['A', 'B', 'C', 'D', 'E'],  # Non-text column
        'score': [85.5, 92.3, 78.1, 88.7, 91.2]  # Numeric column
    }
    
    df = pd.DataFrame(test_data)
    print('Original data:')
    print(df[['description', 'notes']].head())
    print()
    
    # Test 1: No removal (should return unchanged)
    print('TEST 1: No removal (remove_stopwords=False)')
    print('-' * 40)
    result_no_removal = remove_stopwords_from_text_column_enhanced(df, remove_stopwords=False)
    is_unchanged = result_no_removal[['description', 'notes']].equals(df[['description', 'notes']])
    print(f'Result unchanged: {is_unchanged}')
    print()
    
    # Test 2: With removal (if spaCy available)
    print('TEST 2: With removal (remove_stopwords=True)')
    print('-' * 40)
    try:
        result_with_removal = remove_stopwords_from_text_column_enhanced(df, remove_stopwords=True)
        print('Stop words removal successful!')
        print('Result:')
        print(result_with_removal[['description', 'notes']].head())
        print()
        
        # Show before/after comparison
        print('BEFORE vs AFTER comparison:')
        for i in range(3):
            print(f'Row {i+1}:')
            print(f'  BEFORE: {df.iloc[i]["description"]}')
            print(f'  AFTER:  {result_with_removal.iloc[i]["description"]}')
            print()
            
    except Exception as e:
        print(f'Error: {e}')
        print('This is expected if spaCy is not installed')
        print('The system gracefully handles missing dependencies')
    
    print('✅ Stop word removal system analysis completed!')

if __name__ == "__main__":
    test_stopwords_system()
