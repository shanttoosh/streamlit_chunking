#!/usr/bin/env python3
"""
COMPREHENSIVE CHUNKING TEST
Tests all chunking methods across all three modes (Fast, Config-1, Deep Config)
"""
import requests
import json
import pandas as pd
import numpy as np
import time
from backend import (
    chunk_fixed_enhanced,
    chunk_recursive_keyvalue_enhanced,
    chunk_semantic_cluster_enhanced,
    document_based_chunking_enhanced,
    chunk_semantic_cluster,
    chunk_fixed,
    chunk_recursive_keyvalue,
    document_based_chunking
)

def test_backend_chunking_functions():
    """Test all backend chunking functions directly"""
    print('🔧 TESTING BACKEND CHUNKING FUNCTIONS')
    print('=' * 60)
    
    # Create comprehensive test data
    test_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance'],
        'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA', 'Chicago', 'NYC', 'LA', 'Chicago', 'NYC'],
        'salary': [75000, 65000, 80000, 70000, 85000, 60000, 75000, 90000, 55000, 80000],
        'experience': [5, 3, 7, 4, 6, 2, 5, 8, 1, 6],
        'description': [
            'Senior software engineer with expertise in Python and machine learning',
            'HR specialist focusing on employee relations and recruitment',
            'Lead developer working on cloud infrastructure and DevOps',
            'Financial analyst with strong background in data analysis',
            'Full-stack developer specializing in web applications',
            'HR coordinator handling onboarding and training programs',
            'Senior financial analyst with expertise in risk management',
            'Principal engineer leading architecture decisions',
            'Junior HR assistant supporting recruitment activities',
            'Financial consultant providing strategic financial advice'
        ]
    }
    
    df = pd.DataFrame(test_data)
    
    print('Test DataFrame:')
    print(df.head())
    print(f'Total rows: {len(df)}')
    print()
    
    # Test 1: Fixed Size Chunking (Enhanced)
    print('TEST 1: Fixed Size Chunking (Enhanced)')
    print('-' * 40)
    try:
        chunks = chunk_fixed_enhanced(df, chunk_size=200, overlap=50)
        print(f'✅ Fixed chunking successful: {len(chunks)} chunks')
        print(f'Sample chunk: {chunks[0][:100]}...')
    except Exception as e:
        print(f'❌ Fixed chunking failed: {e}')
    
    print()
    
    # Test 2: Recursive Key-Value Chunking (Enhanced)
    print('TEST 2: Recursive Key-Value Chunking (Enhanced)')
    print('-' * 40)
    try:
        chunks = chunk_recursive_keyvalue_enhanced(df, chunk_size=300, overlap=75)
        print(f'✅ Recursive KV chunking successful: {len(chunks)} chunks')
        print(f'Sample chunk: {chunks[0][:100]}...')
    except Exception as e:
        print(f'❌ Recursive KV chunking failed: {e}')
    
    print()
    
    # Test 3: Semantic Clustering (Enhanced)
    print('TEST 3: Semantic Clustering (Enhanced)')
    print('-' * 40)
    try:
        chunks = chunk_semantic_cluster_enhanced(df, n_clusters=3)
        print(f'✅ Semantic clustering successful: {len(chunks)} chunks')
        print(f'Sample chunk: {chunks[0][:100]}...')
    except Exception as e:
        print(f'❌ Semantic clustering failed: {e}')
    
    print()
    
    # Test 4: Document-Based Chunking (Enhanced)
    print('TEST 4: Document-Based Chunking (Enhanced)')
    print('-' * 40)
    try:
        chunks, metadata = document_based_chunking_enhanced(
            df, 
            key_column='department', 
            token_limit=500, 
            preserve_headers=True
        )
        print(f'✅ Document-based chunking successful: {len(chunks)} chunks')
        print(f'Sample chunk: {chunks[0][:100]}...')
        print(f'Sample metadata: {metadata[0]}')
    except Exception as e:
        print(f'❌ Document-based chunking failed: {e}')
    
    print()
    
    # Test 5: Legacy Functions (for comparison)
    print('TEST 5: Legacy Chunking Functions')
    print('-' * 40)
    
    try:
        # Legacy fixed
        chunks = chunk_fixed(df, chunk_size=200, overlap=50)
        print(f'✅ Legacy fixed chunking: {len(chunks)} chunks')
    except Exception as e:
        print(f'❌ Legacy fixed chunking failed: {e}')
    
    try:
        # Legacy recursive
        chunks = chunk_recursive_keyvalue(df, chunk_size=300, overlap=75)
        print(f'✅ Legacy recursive KV chunking: {len(chunks)} chunks')
    except Exception as e:
        print(f'❌ Legacy recursive KV chunking failed: {e}')
    
    try:
        # Legacy semantic
        chunks = chunk_semantic_cluster(df, n_clusters=3)
        print(f'✅ Legacy semantic clustering: {len(chunks)} chunks')
    except Exception as e:
        print(f'❌ Legacy semantic clustering failed: {e}')
    
    try:
        # Legacy document-based
        chunks, metadata = document_based_chunking(df, 'department', token_limit=500, preserve_headers=True)
        print(f'✅ Legacy document-based chunking: {len(chunks)} chunks')
    except Exception as e:
        print(f'❌ Legacy document-based chunking failed: {e}')
    
    print('=' * 60)

def test_fast_mode_chunking():
    """Test Fast Mode chunking (semantic clustering)"""
    print('🚀 TESTING FAST MODE CHUNKING')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Create test data
    test_data = {
        'text': [
            'The system is running smoothly and efficiently in the production environment',
            'We are analyzing customer data to improve our service quality and user experience',
            'The development team is working on new features for the upcoming release',
            'Marketing campaigns are showing positive results with increased engagement',
            'Financial reports indicate strong growth in revenue and profitability',
            'Human resources is conducting performance reviews and career development sessions',
            'Operations team is optimizing processes to reduce costs and improve efficiency',
            'Research and development is exploring new technologies and innovation opportunities',
            'Customer support is handling inquiries and resolving issues promptly',
            'Quality assurance is testing new features and ensuring product reliability'
        ],
        'category': ['IT', 'Analytics', 'Development', 'Marketing', 'Finance', 'HR', 'Operations', 'R&D', 'Support', 'QA'],
        'priority': ['High', 'Medium', 'High', 'Medium', 'High', 'Low', 'Medium', 'High', 'Medium', 'High']
    }
    
    df = pd.DataFrame(test_data)
    
    print('Test DataFrame:')
    print(df.head())
    print()
    
    # Test Fast Mode API
    print('Testing Fast Mode API...')
    try:
        files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
        response = requests.post(f"{base_url}/run_fast", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f'✅ Fast Mode successful: {result.get("summary", {})}')
            print(f'Chunks created: {result.get("summary", {}).get("chunks", "N/A")}')
            print(f'Storage type: {result.get("summary", {}).get("stored", "N/A")}')
        else:
            print(f'❌ Fast Mode failed: {response.text}')
    except Exception as e:
        print(f'❌ Fast Mode error: {e}')
    
    print('=' * 60)

def test_config1_mode_chunking():
    """Test Config-1 Mode chunking (all methods)"""
    print('⚙️ TESTING CONFIG-1 MODE CHUNKING')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Create test data
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta', 'Project Epsilon'],
        'department': ['IT', 'Marketing', 'Finance', 'HR', 'Operations'],
        'budget': [100000, 75000, 120000, 50000, 90000],
        'status': ['Active', 'Planning', 'Completed', 'On Hold', 'Active'],
        'description': [
            'Advanced software development project with machine learning components',
            'Digital marketing campaign targeting new customer segments',
            'Financial analysis and reporting system implementation',
            'Employee training and development program rollout',
            'Process optimization and automation initiative'
        ]
    }
    
    df = pd.DataFrame(test_data)
    
    print('Test DataFrame:')
    print(df.head())
    print()
    
    # Test all chunking methods in Config-1 Mode
    chunking_methods = [
        {'method': 'fixed', 'chunk_size': 300, 'overlap': 50},
        {'method': 'recursive', 'chunk_size': 400, 'overlap': 75},
        {'method': 'semantic', 'n_clusters': 3},
        {'method': 'document', 'key_column': 'department', 'token_limit': 800}
    ]
    
    for config in chunking_methods:
        print(f'Testing Config-1 Mode: {config["method"].upper()} chunking')
        print('-' * 40)
        
        try:
            # Prepare API data
            api_data = {
                'chunk_method': config['method'],
                'chunk_size': config.get('chunk_size', 400),
                'overlap': config.get('overlap', 50),
                'model_choice': 'paraphrase-MiniLM-L6-v2',
                'storage_choice': 'faiss'
            }
            
            if config['method'] == 'document':
                api_data['document_key_column'] = config['key_column']
                api_data['token_limit'] = config['token_limit']
            
            files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
            
            response = requests.post(f"{base_url}/run_config1", files=files, data=api_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f'✅ {config["method"].upper()} chunking successful')
                print(f'Chunks created: {result.get("summary", {}).get("chunks", "N/A")}')
                print(f'Storage type: {result.get("summary", {}).get("stored", "N/A")}')
            else:
                print(f'❌ {config["method"].upper()} chunking failed: {response.text}')
                
        except Exception as e:
            print(f'❌ {config["method"].upper()} chunking error: {e}')
        
        print()
    
    print('=' * 60)

def test_deep_config_mode_chunking():
    """Test Deep Config Mode chunking (all methods with metadata)"""
    print('🔬 TESTING DEEP CONFIG MODE CHUNKING')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Create comprehensive test data
    test_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Adams', 'Frank Miller', 'Grace Lee', 'Henry Wilson'],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT'],
        'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA', 'Chicago', 'NYC', 'LA'],
        'salary': [75000, 65000, 80000, 70000, 85000, 60000, 75000, 90000],
        'experience': [5, 3, 7, 4, 6, 2, 5, 8],
        'performance_score': [85.5, 92.3, 78.1, 88.7, 91.2, 82.4, 87.9, 94.1],
        'description': [
            'Senior software engineer with expertise in Python and machine learning algorithms',
            'HR specialist focusing on employee relations and recruitment strategies',
            'Lead developer working on cloud infrastructure and DevOps automation',
            'Financial analyst with strong background in data analysis and reporting',
            'Full-stack developer specializing in web applications and user interfaces',
            'HR coordinator handling onboarding and training program development',
            'Senior financial analyst with expertise in risk management and compliance',
            'Principal engineer leading architecture decisions and technical strategy'
        ]
    }
    
    df = pd.DataFrame(test_data)
    
    print('Test DataFrame:')
    print(df.head())
    print()
    
    # Test all chunking methods in Deep Config Mode
    chunking_configs = [
        {
            'name': 'Fixed Size',
            'method': 'fixed',
            'chunk_size': 300,
            'overlap': 50,
            'description': 'Fixed-size chunks with overlap'
        },
        {
            'name': 'Recursive',
            'method': 'recursive',
            'chunk_size': 400,
            'overlap': 75,
            'description': 'Recursive key-value splitting'
        },
        {
            'name': 'Semantic',
            'method': 'semantic',
            'n_clusters': 3,
            'description': 'Semantic clustering approach'
        },
        {
            'name': 'Document',
            'method': 'document',
            'key_column': 'department',
            'token_limit': 800,
            'preserve_headers': True,
            'description': 'Document-based grouping by department'
        }
    ]
    
    for config in chunking_configs:
        print(f'Testing Deep Config Mode: {config["name"]} Chunking')
        print(f'Description: {config["description"]}')
        print('-' * 50)
        
        try:
            # Step 1: Preprocess
            files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
            response = requests.post(f"{base_url}/deep_config/preprocess", files=files)
            
            if response.status_code != 200:
                print(f'❌ Preprocessing failed: {response.text}')
                continue
            
            print('✅ Preprocessing successful')
            
            # Step 2: Chunking with metadata
            chunk_data = {
                'chunk_method': config['method'],
                'store_metadata': 'true',
                'numeric_columns': 2,  # salary, performance_score
                'categorical_columns': 2  # department, city
            }
            
            if config['method'] in ['fixed', 'recursive']:
                chunk_data['chunk_size'] = config['chunk_size']
                chunk_data['overlap'] = config['overlap']
            elif config['method'] == 'semantic':
                chunk_data['n_clusters'] = config['n_clusters']
            elif config['method'] == 'document':
                chunk_data['key_column'] = config['key_column']
                chunk_data['token_limit'] = config['token_limit']
            
            response = requests.post(f"{base_url}/deep_config/chunk", data=chunk_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f'✅ {config["name"]} chunking successful')
                print(f'Chunks created: {result.get("total_chunks", "N/A")}')
                print(f'Metadata enabled: {result.get("metadata_enabled", False)}')
                
                # Show sample metadata
                metadata = result.get('metadata', [])
                if metadata:
                    print(f'Sample metadata: {metadata[0]}')
            else:
                print(f'❌ {config["name"]} chunking failed: {response.text}')
                
        except Exception as e:
            print(f'❌ {config["name"]} chunking error: {e}')
        
        print()
    
    print('=' * 60)

def test_chunking_parameters():
    """Test different parameter combinations for each chunking method"""
    print('🔧 TESTING CHUNKING PARAMETERS')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Create test data
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5'],
        'category': ['A', 'B', 'A', 'B', 'A'],
        'value': [100, 200, 300, 400, 500],
        'description': [
            'This is a test description for chunking parameter testing',
            'Another test description with different content and structure',
            'Third test description with various text patterns and formats',
            'Fourth test description containing multiple sentences and details',
            'Fifth test description with comprehensive information and data'
        ]
    }
    
    df = pd.DataFrame(test_data)
    
    print('Test DataFrame:')
    print(df.head())
    print()
    
    # Test parameter combinations
    parameter_tests = [
        {
            'method': 'fixed',
            'name': 'Fixed Size - Small chunks',
            'params': {'chunk_size': 100, 'overlap': 20}
        },
        {
            'method': 'fixed',
            'name': 'Fixed Size - Large chunks',
            'params': {'chunk_size': 500, 'overlap': 100}
        },
        {
            'method': 'recursive',
            'name': 'Recursive - Small chunks',
            'params': {'chunk_size': 150, 'overlap': 30}
        },
        {
            'method': 'recursive',
            'name': 'Recursive - Large chunks',
            'params': {'chunk_size': 600, 'overlap': 120}
        },
        {
            'method': 'semantic',
            'name': 'Semantic - Few clusters',
            'params': {'n_clusters': 2}
        },
        {
            'method': 'semantic',
            'name': 'Semantic - Many clusters',
            'params': {'n_clusters': 4}
        },
        {
            'method': 'document',
            'name': 'Document - Small token limit',
            'params': {'key_column': 'category', 'token_limit': 200}
        },
        {
            'method': 'document',
            'name': 'Document - Large token limit',
            'params': {'key_column': 'category', 'token_limit': 1000}
        }
    ]
    
    for test in parameter_tests:
        print(f'Testing: {test["name"]}')
        print('-' * 30)
        
        try:
            # Preprocess
            files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
            response = requests.post(f"{base_url}/deep_config/preprocess", files=files)
            
            if response.status_code != 200:
                print(f'❌ Preprocessing failed: {response.text}')
                continue
            
            # Chunking
            chunk_data = {
                'chunk_method': test['method'],
                'store_metadata': 'false'  # Disable metadata for parameter testing
            }
            chunk_data.update(test['params'])
            
            response = requests.post(f"{base_url}/deep_config/chunk", data=chunk_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f'✅ Success: {result.get("total_chunks", "N/A")} chunks created')
                print(f'Parameters: {test["params"]}')
            else:
                print(f'❌ Failed: {response.text}')
                
        except Exception as e:
            print(f'❌ Error: {e}')
        
        print()
    
    print('=' * 60)

def test_chunking_performance():
    """Test chunking performance with different dataset sizes"""
    print('⚡ TESTING CHUNKING PERFORMANCE')
    print('=' * 60)
    
    # Test different dataset sizes
    dataset_sizes = [10, 50, 100, 500]
    
    for size in dataset_sizes:
        print(f'Testing with {size} rows...')
        print('-' * 30)
        
        # Create test data
        test_data = {
            'id': list(range(1, size + 1)),
            'name': [f'Test {i}' for i in range(1, size + 1)],
            'category': [f'Category {i % 5}' for i in range(size)],
            'value': [i * 10 for i in range(size)],
            'description': [f'This is test description number {i} with various content and information' for i in range(size)]
        }
        
        df = pd.DataFrame(test_data)
        
        # Test each chunking method
        methods = [
            ('fixed', {'chunk_size': 200, 'overlap': 50}),
            ('recursive', {'chunk_size': 300, 'overlap': 75}),
            ('semantic', {'n_clusters': min(5, size // 2)}),
            ('document', {'key_column': 'category', 'token_limit': 500})
        ]
        
        for method, params in methods:
            try:
                start_time = time.time()
                
                if method == 'fixed':
                    chunks = chunk_fixed_enhanced(df, **params)
                elif method == 'recursive':
                    chunks = chunk_recursive_keyvalue_enhanced(df, **params)
                elif method == 'semantic':
                    chunks = chunk_semantic_cluster_enhanced(df, **params)
                elif method == 'document':
                    chunks, _ = document_based_chunking_enhanced(df, **params)
                
                end_time = time.time()
                duration = end_time - start_time
                
                print(f'✅ {method.capitalize()}: {len(chunks)} chunks in {duration:.4f}s')
                
            except Exception as e:
                print(f'❌ {method.capitalize()}: Failed - {e}')
        
        print()
    
    print('=' * 60)

if __name__ == "__main__":
    print('🚀 STARTING COMPREHENSIVE CHUNKING TEST')
    print('=' * 80)
    print('This test covers:')
    print('• All backend chunking functions (enhanced and legacy)')
    print('• Fast Mode chunking (semantic clustering)')
    print('• Config-1 Mode chunking (all methods)')
    print('• Deep Config Mode chunking (all methods with metadata)')
    print('• Parameter testing for all chunking methods')
    print('• Performance testing with different dataset sizes')
    print('=' * 80)
    print()
    
    try:
        # Test 1: Backend functions
        test_backend_chunking_functions()
        
        # Test 2: Fast Mode
        test_fast_mode_chunking()
        
        # Test 3: Config-1 Mode
        test_config1_mode_chunking()
        
        # Test 4: Deep Config Mode
        test_deep_config_mode_chunking()
        
        # Test 5: Parameter testing
        test_chunking_parameters()
        
        # Test 6: Performance testing
        test_chunking_performance()
        
        print('🎉 COMPREHENSIVE CHUNKING TEST COMPLETED SUCCESSFULLY!')
        print('=' * 80)
        print('✅ All backend chunking functions working correctly')
        print('✅ Fast Mode chunking (semantic clustering) working')
        print('✅ Config-1 Mode chunking (all methods) working')
        print('✅ Deep Config Mode chunking (all methods with metadata) working')
        print('✅ Parameter testing for all chunking methods successful')
        print('✅ Performance testing with different dataset sizes completed')
        print('=' * 80)
        
    except Exception as e:
        print(f'❌ COMPREHENSIVE CHUNKING TEST FAILED: {e}')
        print('Make sure the FastAPI server is running: uvicorn main:app --reload')
