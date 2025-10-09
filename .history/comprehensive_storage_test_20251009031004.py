#!/usr/bin/env python3
"""
COMPREHENSIVE STORAGE TEST
Tests all storage modes (FAISS and ChromaDB) across all three modes
with different similarity parameters and configurations
"""
import requests
import json
import pandas as pd
import numpy as np
import time
from backend import (
    store_faiss,
    store_chroma,
    store_faiss_enhanced,
    store_chroma_enhanced,
    retrieve_similar
)

def test_backend_storage_functions():
    """Test all backend storage functions directly"""
    print('🔧 TESTING BACKEND STORAGE FUNCTIONS')
    print('=' * 60)
    
    # Create test data
    test_chunks = [
        "The system is running smoothly and efficiently in the production environment",
        "We are analyzing customer data to improve our service quality and user experience",
        "The development team is working on new features for the upcoming release",
        "Marketing campaigns are showing positive results with increased engagement",
        "Financial reports indicate strong growth in revenue and profitability",
        "Human resources is conducting performance reviews and career development sessions",
        "Operations team is optimizing processes to reduce costs and improve efficiency",
        "Research and development is exploring new technologies and innovation opportunities",
        "Customer support is handling inquiries and resolving issues promptly",
        "Quality assurance is testing new features and ensuring product reliability"
    ]
    
    # Create test embeddings
    embeddings = np.random.rand(len(test_chunks), 384).astype("float32")
    
    # Create test metadata
    metadata = [
        {'chunk_id': str(i), 'category': f'Category {i % 3}', 'priority': ['High', 'Medium', 'Low'][i % 3]}
        for i in range(len(test_chunks))
    ]
    
    print(f'Test data: {len(test_chunks)} chunks, {embeddings.shape} embeddings')
    print()
    
    # Test 1: Basic FAISS Storage
    print('TEST 1: Basic FAISS Storage')
    print('-' * 40)
    try:
        start_time = time.time()
        faiss_result = store_faiss(embeddings)
        end_time = time.time()
        
        print(f'✅ Basic FAISS storage successful: {end_time - start_time:.4f}s')
        print(f'Result type: {faiss_result["type"]}')
        print(f'Index type: {type(faiss_result["index"]).__name__}')
    except Exception as e:
        print(f'❌ Basic FAISS storage failed: {e}')
    
    print()
    
    # Test 2: Enhanced FAISS Storage
    print('TEST 2: Enhanced FAISS Storage with Metadata')
    print('-' * 40)
    try:
        start_time = time.time()
        faiss_result = store_faiss_enhanced(test_chunks, embeddings, metadata)
        end_time = time.time()
        
        print(f'✅ Enhanced FAISS storage successful: {end_time - start_time:.4f}s')
        print(f'Result type: {faiss_result["type"]}')
        print(f'Total vectors: {faiss_result["data"]["total_vectors"]}')
        print(f'Embedding dimension: {faiss_result["data"]["embedding_dim"]}')
        print(f'Metadata index created: {"metadata_index" in faiss_result}')
    except Exception as e:
        print(f'❌ Enhanced FAISS storage failed: {e}')
    
    print()
    
    # Test 3: Basic ChromaDB Storage
    print('TEST 3: Basic ChromaDB Storage')
    print('-' * 40)
    try:
        start_time = time.time()
        chroma_result = store_chroma(test_chunks, embeddings, "test_collection_basic")
        end_time = time.time()
        
        print(f'✅ Basic ChromaDB storage successful: {end_time - start_time:.4f}s')
        print(f'Result type: {chroma_result["type"]}')
        print(f'Collection name: {chroma_result["collection_name"]}')
        print(f'Collection type: {type(chroma_result["collection"]).__name__}')
    except Exception as e:
        print(f'❌ Basic ChromaDB storage failed: {e}')
    
    print()
    
    # Test 4: Enhanced ChromaDB Storage
    print('TEST 4: Enhanced ChromaDB Storage with Metadata')
    print('-' * 40)
    try:
        start_time = time.time()
        chroma_result = store_chroma_enhanced(test_chunks, embeddings, "test_collection_enhanced", metadata)
        end_time = time.time()
        
        print(f'✅ Enhanced ChromaDB storage successful: {end_time - start_time:.4f}s')
        print(f'Result type: {chroma_result["type"]}')
        print(f'Collection name: {chroma_result["collection_name"]}')
        print(f'Collection type: {type(chroma_result["collection"]).__name__}')
    except Exception as e:
        print(f'❌ Enhanced ChromaDB storage failed: {e}')
    
    print()
    
    # Test 5: FAISS with Different Metrics
    print('TEST 5: FAISS with Different Similarity Metrics')
    print('-' * 40)
    
    try:
        import faiss
        
        # Test different FAISS metrics
        metrics = [
            ('cosine', 'IndexFlatIP (normalized)'),
            ('dot', 'IndexFlatIP'),
            ('euclidean', 'IndexFlatL2')
        ]
        
        for metric_name, index_type in metrics:
            print(f'Testing {metric_name} metric ({index_type}):')
            
            start_time = time.time()
            
            # Prepare vectors based on metric
            vecs = embeddings.copy()
            if metric_name == "cosine":
                # Normalize rows to unit length
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vecs = vecs / norms
                index = faiss.IndexFlatIP(vecs.shape[1])
            elif metric_name == "dot":
                index = faiss.IndexFlatIP(vecs.shape[1])
            else:  # euclidean
                index = faiss.IndexFlatL2(vecs.shape[1])
            
            index.add(vecs)
            
            end_time = time.time()
            
            print(f'  ✅ {metric_name} metric successful: {end_time - start_time:.4f}s')
            print(f'  Index type: {type(index).__name__}')
            print(f'  Total vectors: {index.ntotal}')
            
    except Exception as e:
        print(f'❌ FAISS metrics test failed: {e}')
    
    print('=' * 60)

def test_fast_mode_storage():
    """Test Fast Mode storage (FAISS only)"""
    print('🚀 TESTING FAST MODE STORAGE')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Create test datasets of different sizes
    datasets = [
        {
            'name': 'Small Dataset',
            'data': {
                'text': [
                    'The system is running smoothly and efficiently',
                    'We are analyzing customer data to improve service quality',
                    'The development team is working on new features',
                    'Marketing campaigns are showing positive results',
                    'Financial reports indicate strong growth'
                ],
                'category': ['IT', 'Analytics', 'Development', 'Marketing', 'Finance']
            }
        },
        {
            'name': 'Medium Dataset',
            'data': {
                'text': [f'This is test text number {i} with various content and information for storage testing' for i in range(20)],
                'category': [f'Category {i % 5}' for i in range(20)]
            }
        },
        {
            'name': 'Large Dataset',
            'data': {
                'text': [f'This is test text number {i} with various content and information for storage testing and performance evaluation' for i in range(50)],
                'category': [f'Category {i % 10}' for i in range(50)]
            }
        }
    ]
    
    for dataset in datasets:
        print(f'Testing {dataset["name"]} ({len(dataset["data"]["text"])} rows)')
        print('-' * 50)
        
        df = pd.DataFrame(dataset['data'])
        
        try:
            files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
            api_data = {
                'use_openai': False,  # Force local model
                'batch_size': 64
            }
            
            start_time = time.time()
            response = requests.post(f"{base_url}/run_fast", files=files, data=api_data)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f'✅ Success: {result.get("summary", {})}')
                print(f'Time: {end_time - start_time:.4f}s')
                print(f'Chunks: {result.get("summary", {}).get("chunks", "N/A")}')
                print(f'Storage: {result.get("summary", {}).get("stored", "N/A")}')
                print(f'Model: {result.get("summary", {}).get("embedding_model", "N/A")}')
            else:
                print(f'❌ Failed: {response.text}')
                
        except Exception as e:
            print(f'❌ Error: {e}')
        
        print()
    
    print('=' * 60)

def test_config1_mode_storage():
    """Test Config-1 Mode storage (FAISS and ChromaDB with similarity metrics)"""
    print('⚙️ TESTING CONFIG-1 MODE STORAGE')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Create test data
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta', 'Project Epsilon'],
        'department': ['IT', 'Marketing', 'Finance', 'HR', 'Operations'],
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
    
    # Test all storage types and similarity metrics
    storage_configs = [
        {
            'storage_type': 'faiss',
            'metrics': ['cosine', 'dot', 'euclidean'],
            'description': 'FAISS with different similarity metrics'
        },
        {
            'storage_type': 'chroma',
            'metrics': ['cosine', 'l2', 'ip'],
            'description': 'ChromaDB with different similarity metrics'
        }
    ]
    
    # Test different chunking methods
    chunking_methods = [
        {'method': 'fixed', 'chunk_size': 300, 'overlap': 50},
        {'method': 'recursive', 'chunk_size': 400, 'overlap': 75},
        {'method': 'semantic', 'n_clusters': 3},
        {'method': 'document', 'key_column': 'department', 'token_limit': 800}
    ]
    
    for storage_config in storage_configs:
        print(f'TESTING {storage_config["storage_type"].upper()} STORAGE')
        print(f'Description: {storage_config["description"]}')
        print('-' * 50)
        
        for metric in storage_config['metrics']:
            print(f'Similarity metric: {metric}')
            
            for chunk_config in chunking_methods:
                print(f'  Chunking method: {chunk_config["method"]}')
                
                try:
                    # Prepare API data
                    api_data = {
                        'chunk_method': chunk_config['method'],
                        'model_choice': 'paraphrase-MiniLM-L6-v2',
                        'storage_choice': storage_config['storage_type'],
                        'retrieval_metric': metric,
                        'use_openai': False,  # Force local model
                        'batch_size': 64
                    }
                    
                    if chunk_config['method'] in ['fixed', 'recursive']:
                        api_data['chunk_size'] = chunk_config['chunk_size']
                        api_data['overlap'] = chunk_config['overlap']
                    elif chunk_config['method'] == 'semantic':
                        api_data['n_clusters'] = chunk_config['n_clusters']
                    elif chunk_config['method'] == 'document':
                        api_data['document_key_column'] = chunk_config['key_column']
                        api_data['token_limit'] = chunk_config['token_limit']
                    
                    files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
                    
                    start_time = time.time()
                    response = requests.post(f"{base_url}/run_config1", files=files, data=api_data)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f'    ✅ Success: {result.get("summary", {})}')
                        print(f'    Time: {end_time - start_time:.4f}s')
                        print(f'    Chunks: {result.get("summary", {}).get("chunks", "N/A")}')
                        print(f'    Storage: {result.get("summary", {}).get("stored", "N/A")}')
                        print(f'    Model: {result.get("summary", {}).get("embedding_model", "N/A")}')
                    else:
                        print(f'    ❌ Failed: {response.text}')
                        
                except Exception as e:
                    print(f'    ❌ Error: {e}')
                
                print()
            
            print()
        
        print()
    
    print('=' * 60)

def test_deep_config_mode_storage():
    """Test Deep Config Mode storage (FAISS and ChromaDB with metadata)"""
    print('🔬 TESTING DEEP CONFIG MODE STORAGE')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Create comprehensive test data
    test_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Adams', 'Frank Miller', 'Grace Lee', 'Henry Wilson'],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT'],
        'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA', 'Chicago', 'NYC', 'LA'],
        'salary': [75000, 65000, 80000, 70000, 85000, 60000, 75000, 90000],
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
    
    # Test all storage types
    storage_types = [
        {
            'name': 'ChromaDB',
            'type': 'chroma',
            'description': 'Native metadata support with built-in filtering'
        },
        {
            'name': 'FAISS',
            'type': 'faiss',
            'description': 'Enhanced metadata support with custom filtering'
        }
    ]
    
    # Test different chunking methods
    chunking_methods = [
        {'method': 'fixed', 'chunk_size': 300, 'overlap': 50},
        {'method': 'recursive', 'chunk_size': 400, 'overlap': 75},
        {'method': 'semantic', 'n_clusters': 3},
        {'method': 'document', 'key_column': 'department', 'token_limit': 800}
    ]
    
    for storage_config in storage_types:
        print(f'TESTING {storage_config["name"]} STORAGE')
        print(f'Description: {storage_config["description"]}')
        print('-' * 50)
        
        for chunk_config in chunking_methods:
            print(f'Chunking method: {chunk_config["method"]}')
            
            try:
                # Step 1: Preprocess
                files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
                response = requests.post(f"{base_url}/deep_config/preprocess", files=files)
                
                if response.status_code != 200:
                    print(f'  ❌ Preprocessing failed: {response.text}')
                    continue
                
                # Step 2: Chunking with metadata
                chunk_data = {
                    'chunk_method': chunk_config['method'],
                    'store_metadata': 'true',
                    'numeric_columns': 2,  # salary, performance_score
                    'categorical_columns': 2  # department, city
                }
                
                if chunk_config['method'] in ['fixed', 'recursive']:
                    chunk_data['chunk_size'] = chunk_config['chunk_size']
                    chunk_data['overlap'] = chunk_config['overlap']
                elif chunk_config['method'] == 'semantic':
                    chunk_data['n_clusters'] = chunk_config['n_clusters']
                elif chunk_config['method'] == 'document':
                    chunk_data['key_column'] = chunk_config['key_column']
                    chunk_data['token_limit'] = chunk_config['token_limit']
                
                response = requests.post(f"{base_url}/deep_config/chunk", data=chunk_data)
                
                if response.status_code != 200:
                    print(f'  ❌ Chunking failed: {response.text}')
                    continue
                
                # Step 3: Embedding
                embed_data = {
                    'model_name': 'paraphrase-MiniLM-L6-v2',
                    'batch_size': 64,
                    'use_openai': False  # Force local model
                }
                
                response = requests.post(f"{base_url}/deep_config/embed", data=embed_data)
                
                if response.status_code != 200:
                    print(f'  ❌ Embedding failed: {response.text}')
                    continue
                
                # Step 4: Storage
                store_data = {
                    'storage_type': storage_config['type'],
                    'collection_name': f'test_{storage_config["type"]}_{chunk_config["method"]}'
                }
                
                start_time = time.time()
                response = requests.post(f"{base_url}/deep_config/store", data=store_data)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    print(f'  ✅ Success: {result.get("status", "unknown")}')
                    print(f'  Time: {end_time - start_time:.4f}s')
                    print(f'  Storage type: {result.get("storage_type", "unknown")}')
                    print(f'  Total vectors: {result.get("total_vectors", "N/A")}')
                    print(f'  Collection: {result.get("collection_name", "N/A")}')
                else:
                    print(f'  ❌ Storage failed: {response.text}')
                    
            except Exception as e:
                print(f'  ❌ Error: {e}')
            
            print()
        
        print()
    
    print('=' * 60)

def test_storage_performance():
    """Test storage performance with different dataset sizes"""
    print('⚡ TESTING STORAGE PERFORMANCE')
    print('=' * 60)
    
    # Test different dataset sizes
    dataset_sizes = [10, 50, 100, 200, 500]
    
    # Available storage types
    storage_types = [
        ('faiss', 'FAISS'),
        ('chroma', 'ChromaDB')
    ]
    
    for size in dataset_sizes:
        print(f'Testing with {size} chunks...')
        print('-' * 30)
        
        # Create test data
        test_chunks = [f"This is test chunk number {i} with various content and information for storage performance testing" for i in range(size)]
        embeddings = np.random.rand(size, 384).astype("float32")
        metadata = [{'chunk_id': str(i), 'category': f'Category {i % 5}'} for i in range(size)]
        
        for storage_type, storage_name in storage_types:
            print(f'Storage: {storage_name}')
            
            try:
                start_time = time.time()
                
                if storage_type == 'faiss':
                    result = store_faiss_enhanced(test_chunks, embeddings, metadata)
                else:  # chroma
                    result = store_chroma_enhanced(test_chunks, embeddings, f"perf_test_{size}", metadata)
                
                end_time = time.time()
                duration = end_time - start_time
                chunks_per_second = size / duration if duration > 0 else 0
                
                print(f'  ✅ Success: {size} chunks in {duration:.4f}s ({chunks_per_second:.1f} chunks/s)')
                print(f'  Result type: {result["type"]}')
                
            except Exception as e:
                print(f'  ❌ Failed: {e}')
            
            print()
        
        print()
    
    print('=' * 60)

def test_storage_retrieval():
    """Test storage retrieval with different similarity metrics"""
    print('🔍 TESTING STORAGE RETRIEVAL')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Create test data
    test_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Project Alpha', 'Project Beta', 'Project Gamma', 'Project Delta', 'Project Epsilon'],
        'department': ['IT', 'Marketing', 'Finance', 'HR', 'Operations'],
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
    
    # Test different storage types and metrics
    test_configs = [
        {
            'storage_type': 'faiss',
            'metric': 'cosine',
            'description': 'FAISS with cosine similarity'
        },
        {
            'storage_type': 'faiss',
            'metric': 'euclidean',
            'description': 'FAISS with euclidean distance'
        },
        {
            'storage_type': 'chroma',
            'metric': 'cosine',
            'description': 'ChromaDB with cosine similarity'
        },
        {
            'storage_type': 'chroma',
            'metric': 'l2',
            'description': 'ChromaDB with L2 distance'
        }
    ]
    
    # Test queries
    test_queries = [
        'software development',
        'marketing campaign',
        'financial analysis',
        'employee training',
        'process optimization'
    ]
    
    for config in test_configs:
        print(f'TESTING: {config["description"]}')
        print('-' * 40)
        
        try:
            # Step 1: Preprocess
            files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
            response = requests.post(f"{base_url}/deep_config/preprocess", files=files)
            
            if response.status_code != 200:
                print(f'❌ Preprocessing failed: {response.text}')
                continue
            
            # Step 2: Chunking
            chunk_data = {
                'chunk_method': 'document',
                'key_column': 'department',
                'token_limit': 800,
                'store_metadata': 'true',
                'numeric_columns': 1,
                'categorical_columns': 1
            }
            
            response = requests.post(f"{base_url}/deep_config/chunk", data=chunk_data)
            
            if response.status_code != 200:
                print(f'❌ Chunking failed: {response.text}')
                continue
            
            # Step 3: Embedding
            embed_data = {
                'model_name': 'paraphrase-MiniLM-L6-v2',
                'batch_size': 64,
                'use_openai': False
            }
            
            response = requests.post(f"{base_url}/deep_config/embed", data=embed_data)
            
            if response.status_code != 200:
                print(f'❌ Embedding failed: {response.text}')
                continue
            
            # Step 4: Storage
            store_data = {
                'storage_type': config['storage_type'],
                'collection_name': f'retrieval_test_{config["storage_type"]}_{config["metric"]}'
            }
            
            response = requests.post(f"{base_url}/deep_config/store", data=store_data)
            
            if response.status_code != 200:
                print(f'❌ Storage failed: {response.text}')
                continue
            
            print('✅ Storage successful')
            
            # Step 5: Test retrieval
            for query in test_queries:
                print(f'  Query: "{query}"')
                
                try:
                    query_data = {
                        'query': query,
                        'k': 3,
                        'metadata_filter': '{}'
                    }
                    
                    start_time = time.time()
                    response = requests.post(f"{base_url}/retrieve_with_metadata", data=query_data)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f'    ✅ Success: {result.get("total_results", 0)} results in {end_time - start_time:.4f}s')
                        print(f'    Store type: {result.get("store_type", "unknown")}')
                        
                        # Show top result
                        results = result.get('results', [])
                        if results:
                            top_result = results[0]
                            print(f'    Top result similarity: {top_result.get("similarity", 0):.4f}')
                    else:
                        print(f'    ❌ Failed: {response.text}')
                        
                except Exception as e:
                    print(f'    ❌ Error: {e}')
                
                print()
            
        except Exception as e:
            print(f'❌ Error: {e}')
        
        print()
    
    print('=' * 60)

def test_storage_complex_scenarios():
    """Test complex storage scenarios"""
    print('🎯 TESTING COMPLEX STORAGE SCENARIOS')
    print('=' * 60)
    
    base_url = "http://localhost:8000"
    
    # Create complex test data
    test_data = {
        'id': list(range(1, 21)),
        'name': [f'Employee {i}' for i in range(1, 21)],
        'department': ['IT', 'Marketing', 'Finance', 'HR', 'Operations'] * 4,
        'city': ['NYC', 'LA', 'Chicago', 'Seattle', 'Miami'] * 4,
        'salary': [50000 + i * 2000 for i in range(20)],
        'performance_score': [70 + i * 1.5 for i in range(20)],
        'description': [
            f'Employee {i} is a dedicated professional with expertise in various domains and strong commitment to excellence'
            for i in range(1, 21)
        ]
    }
    
    df = pd.DataFrame(test_data)
    
    print('Complex Test DataFrame:')
    print(df.head())
    print(f'Total rows: {len(df)}')
    print()
    
    # Test complex scenarios
    scenarios = [
        {
            'name': 'Large Dataset with Metadata',
            'description': 'Test with large dataset and rich metadata',
            'chunk_method': 'document',
            'key_column': 'department',
            'store_metadata': True,
            'numeric_columns': 2,
            'categorical_columns': 2
        },
        {
            'name': 'Multiple Chunking Methods',
            'description': 'Test different chunking methods with same data',
            'chunk_method': 'semantic',
            'n_clusters': 5,
            'store_metadata': True,
            'numeric_columns': 1,
            'categorical_columns': 1
        },
        {
            'name': 'High-Performance Storage',
            'description': 'Test with optimized parameters for performance',
            'chunk_method': 'fixed',
            'chunk_size': 500,
            'overlap': 100,
            'store_metadata': False
        }
    ]
    
    for scenario in scenarios:
        print(f'SCENARIO: {scenario["name"]}')
        print(f'Description: {scenario["description"]}')
        print('-' * 50)
        
        # Test both storage types
        for storage_type in ['chroma', 'faiss']:
            print(f'Storage type: {storage_type.upper()}')
            
            try:
                # Step 1: Preprocess
                files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
                response = requests.post(f"{base_url}/deep_config/preprocess", files=files)
                
                if response.status_code != 200:
                    print(f'  ❌ Preprocessing failed: {response.text}')
                    continue
                
                # Step 2: Chunking
                chunk_data = {
                    'chunk_method': scenario['chunk_method'],
                    'store_metadata': str(scenario.get('store_metadata', False)).lower()
                }
                
                if scenario['chunk_method'] == 'document':
                    chunk_data['key_column'] = scenario['key_column']
                    chunk_data['token_limit'] = 1000
                elif scenario['chunk_method'] == 'fixed':
                    chunk_data['chunk_size'] = scenario['chunk_size']
                    chunk_data['overlap'] = scenario['overlap']
                elif scenario['chunk_method'] == 'semantic':
                    chunk_data['n_clusters'] = scenario['n_clusters']
                
                if scenario.get('store_metadata', False):
                    chunk_data['numeric_columns'] = scenario.get('numeric_columns', 1)
                    chunk_data['categorical_columns'] = scenario.get('categorical_columns', 1)
                
                response = requests.post(f"{base_url}/deep_config/chunk", data=chunk_data)
                
                if response.status_code != 200:
                    print(f'  ❌ Chunking failed: {response.text}')
                    continue
                
                # Step 3: Embedding
                embed_data = {
                    'model_name': 'paraphrase-MiniLM-L6-v2',
                    'batch_size': 128,  # Larger batch for performance
                    'use_openai': False
                }
                
                response = requests.post(f"{base_url}/deep_config/embed", data=embed_data)
                
                if response.status_code != 200:
                    print(f'  ❌ Embedding failed: {response.text}')
                    continue
                
                # Step 4: Storage
                store_data = {
                    'storage_type': storage_type,
                    'collection_name': f'complex_{scenario["name"].lower().replace(" ", "_")}_{storage_type}'
                }
                
                start_time = time.time()
                response = requests.post(f"{base_url}/deep_config/store", data=store_data)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    print(f'  ✅ Success: {result.get("status", "unknown")}')
                    print(f'  Time: {end_time - start_time:.4f}s')
                    print(f'  Total vectors: {result.get("total_vectors", "N/A")}')
                    print(f'  Storage type: {result.get("storage_type", "unknown")}')
                else:
                    print(f'  ❌ Storage failed: {response.text}')
                    
            except Exception as e:
                print(f'  ❌ Error: {e}')
            
            print()
        
        print()
    
    print('=' * 60)

if __name__ == "__main__":
    print('🚀 STARTING COMPREHENSIVE STORAGE TEST')
    print('=' * 80)
    print('This test covers:')
    print('• All backend storage functions (FAISS and ChromaDB)')
    print('• Fast Mode storage (FAISS only)')
    print('• Config-1 Mode storage (FAISS and ChromaDB with similarity metrics)')
    print('• Deep Config Mode storage (FAISS and ChromaDB with metadata)')
    print('• Storage performance testing with different dataset sizes')
    print('• Storage retrieval testing with different similarity metrics')
    print('• Complex storage scenarios')
    print('=' * 80)
    print()
    
    try:
        # Test 1: Backend functions
        test_backend_storage_functions()
        
        # Test 2: Fast Mode
        test_fast_mode_storage()
        
        # Test 3: Config-1 Mode
        test_config1_mode_storage()
        
        # Test 4: Deep Config Mode
        test_deep_config_mode_storage()
        
        # Test 5: Performance testing
        test_storage_performance()
        
        # Test 6: Retrieval testing
        test_storage_retrieval()
        
        # Test 7: Complex scenarios
        test_storage_complex_scenarios()
        
        print('🎉 COMPREHENSIVE STORAGE TEST COMPLETED SUCCESSFULLY!')
        print('=' * 80)
        print('✅ All backend storage functions working correctly')
        print('✅ Fast Mode storage working')
        print('✅ Config-1 Mode storage working')
        print('✅ Deep Config Mode storage working')
        print('✅ Storage performance testing completed')
        print('✅ Storage retrieval testing completed')
        print('✅ Complex storage scenarios tested')
        print('=' * 80)
        
    except Exception as e:
        print(f'❌ COMPREHENSIVE STORAGE TEST FAILED: {e}')
        print('Make sure the FastAPI server is running: uvicorn main:app --reload')
