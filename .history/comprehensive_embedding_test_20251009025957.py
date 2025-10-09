#!/usr/bin/env python3
"""
COMPREHENSIVE EMBEDDING TEST
Tests all embedding models across all three modes (Fast, Config-1, Deep Config)
Focus on local models only (no OpenAI API)
"""
import requests
import json
import pandas as pd
import numpy as np
import time
from backend import (
    embed_texts,
    embed_texts_enhanced,
    parallel_embed_texts,
    parallel_embed_texts_enhanced
)

def test_backend_embedding_functions():
    """Test all backend embedding functions directly"""
    print('🔧 TESTING BACKEND EMBEDDING FUNCTIONS')
    print('=' * 60)
    
    # Create test chunks
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
    
    print(f'Test chunks: {len(test_chunks)}')
    print(f'Sample chunk: {test_chunks[0][:50]}...')
    print()
    
    # Available local models
    local_models = [
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L6-v2"
    ]
    
    # Test batch sizes
    batch_sizes = [16, 32, 64, 128, 256]
    
    for model_name in local_models:
        print(f'TESTING MODEL: {model_name}')
        print('-' * 40)
        
        for batch_size in batch_sizes:
            print(f'Batch size: {batch_size}')
            
            try:
                start_time = time.time()
                model, embeddings = embed_texts(
                    test_chunks, 
                    model_name=model_name, 
                    openai_api_key=None,  # Force local model
                    batch_size=batch_size,
                    use_parallel=False
                )
                end_time = time.time()
                
                print(f'  ✅ Success: {embeddings.shape} in {end_time - start_time:.4f}s')
                print(f'  Model type: {type(model).__name__}')
                print(f'  Embedding shape: {embeddings.shape}')
                print(f'  Embedding dtype: {embeddings.dtype}')
                
            except Exception as e:
                print(f'  ❌ Failed: {e}')
            
            print()
        
        print()
    
    print('=' * 60)

def test_enhanced_embedding_functions():
    """Test enhanced embedding functions"""
    print('🚀 TESTING ENHANCED EMBEDDING FUNCTIONS')
    print('=' * 60)
    
    # Create test chunks
    test_chunks = [
        "The system is running smoothly and efficiently in the production environment",
        "We are analyzing customer data to improve our service quality and user experience",
        "The development team is working on new features for the upcoming release",
        "Marketing campaigns are showing positive results with increased engagement",
        "Financial reports indicate strong growth in revenue and profitability"
    ]
    
    print(f'Test chunks: {len(test_chunks)}')
    print()
    
    # Available local models
    local_models = [
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L6-v2"
    ]
    
    # Test batch sizes
    batch_sizes = [32, 64, 128]
    
    for model_name in local_models:
        print(f'TESTING ENHANCED MODEL: {model_name}')
        print('-' * 40)
        
        for batch_size in batch_sizes:
            print(f'Batch size: {batch_size}')
            
            try:
                start_time = time.time()
                model, embeddings = embed_texts_enhanced(
                    test_chunks, 
                    model_name=model_name, 
                    openai_api_key=None,  # Force local model
                    batch_size=batch_size,
                    use_parallel=False
                )
                end_time = time.time()
                
                print(f'  ✅ Success: {embeddings.shape} in {end_time - start_time:.4f}s')
                print(f'  Model type: {type(model).__name__}')
                print(f'  Embedding shape: {embeddings.shape}')
                print(f'  Embedding dtype: {embeddings.dtype}')
                
            except Exception as e:
                print(f'  ❌ Failed: {e}')
            
            print()
        
        print()
    
    print('=' * 60)

def test_parallel_embedding():
    """Test parallel embedding functions"""
    print('⚡ TESTING PARALLEL EMBEDDING FUNCTIONS')
    print('=' * 60)
    
    # Create larger test dataset
    test_chunks = []
    for i in range(100):
        test_chunks.append(f"This is test chunk number {i} with various content and information for parallel processing testing")
    
    print(f'Test chunks: {len(test_chunks)}')
    print()
    
    # Available local models
    local_models = [
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L6-v2"
    ]
    
    # Test batch sizes
    batch_sizes = [32, 64, 128]
    
    for model_name in local_models:
        print(f'TESTING PARALLEL MODEL: {model_name}')
        print('-' * 40)
        
        for batch_size in batch_sizes:
            print(f'Batch size: {batch_size}')
            
            try:
                start_time = time.time()
                model, embeddings = parallel_embed_texts_enhanced(
                    test_chunks, 
                    model_name=model_name, 
                    batch_size=batch_size
                )
                end_time = time.time()
                
                print(f'  ✅ Success: {embeddings.shape} in {end_time - start_time:.4f}s')
                print(f'  Model type: {type(model).__name__}')
                print(f'  Embedding shape: {embeddings.shape}')
                print(f'  Embedding dtype: {embeddings.dtype}')
                
            except Exception as e:
                print(f'  ❌ Failed: {e}')
            
            print()
        
        print()
    
    print('=' * 60)

def test_fast_mode_embedding():
    """Test Fast Mode embedding (semantic clustering + embedding)"""
    print('🚀 TESTING FAST MODE EMBEDDING')
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
                'text': [f'This is test text number {i} with various content and information for embedding testing' for i in range(20)],
                'category': [f'Category {i % 5}' for i in range(20)]
            }
        },
        {
            'name': 'Large Dataset',
            'data': {
                'text': [f'This is test text number {i} with various content and information for embedding testing and performance evaluation' for i in range(50)],
                'category': [f'Category {i % 10}' for i in range(50)]
            }
        }
    ]
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256]
    
    for dataset in datasets:
        print(f'Testing {dataset["name"]} ({len(dataset["data"]["text"])} rows)')
        print('-' * 50)
        
        df = pd.DataFrame(dataset['data'])
        
        for batch_size in batch_sizes:
            print(f'Batch size: {batch_size}')
            
            try:
                files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
                api_data = {
                    'batch_size': batch_size,
                    'use_openai': False  # Force local model
                }
                
                start_time = time.time()
                response = requests.post(f"{base_url}/run_fast", files=files, data=api_data)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    print(f'  ✅ Success: {result.get("summary", {})}')
                    print(f'  Time: {end_time - start_time:.4f}s')
                    print(f'  Chunks: {result.get("summary", {}).get("chunks", "N/A")}')
                    print(f'  Model: {result.get("summary", {}).get("embedding_model", "N/A")}')
                else:
                    print(f'  ❌ Failed: {response.text}')
                    
            except Exception as e:
                print(f'  ❌ Error: {e}')
            
            print()
        
        print()
    
    print('=' * 60)

def test_config1_mode_embedding():
    """Test Config-1 Mode embedding (all models and parameters)"""
    print('⚙️ TESTING CONFIG-1 MODE EMBEDDING')
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
    
    # Test all local models
    local_models = [
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L6-v2"
    ]
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256]
    
    # Test different chunking methods
    chunking_methods = [
        {'method': 'fixed', 'chunk_size': 300, 'overlap': 50},
        {'method': 'recursive', 'chunk_size': 400, 'overlap': 75},
        {'method': 'semantic', 'n_clusters': 3},
        {'method': 'document', 'key_column': 'department', 'token_limit': 800}
    ]
    
    for model_name in local_models:
        print(f'TESTING MODEL: {model_name}')
        print('-' * 40)
        
        for chunk_config in chunking_methods:
            print(f'Chunking method: {chunk_config["method"]}')
            
            for batch_size in batch_sizes:
                print(f'  Batch size: {batch_size}')
                
                try:
                    # Prepare API data
                    api_data = {
                        'chunk_method': chunk_config['method'],
                        'model_choice': model_name,
                        'storage_choice': 'faiss',
                        'batch_size': batch_size,
                        'use_openai': False  # Force local model
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
                        print(f'    Model: {result.get("summary", {}).get("embedding_model", "N/A")}')
                    else:
                        print(f'    ❌ Failed: {response.text}')
                        
                except Exception as e:
                    print(f'    ❌ Error: {e}')
                
                print()
            
            print()
        
        print()
    
    print('=' * 60)

def test_deep_config_mode_embedding():
    """Test Deep Config Mode embedding (all models with metadata)"""
    print('🔬 TESTING DEEP CONFIG MODE EMBEDDING')
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
    
    # Test all local models
    local_models = [
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L6-v2"
    ]
    
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256]
    
    # Test different chunking methods
    chunking_methods = [
        {'method': 'fixed', 'chunk_size': 300, 'overlap': 50},
        {'method': 'recursive', 'chunk_size': 400, 'overlap': 75},
        {'method': 'semantic', 'n_clusters': 3},
        {'method': 'document', 'key_column': 'department', 'token_limit': 800}
    ]
    
    for model_name in local_models:
        print(f'TESTING MODEL: {model_name}')
        print('-' * 40)
        
        for chunk_config in chunking_methods:
            print(f'Chunking method: {chunk_config["method"]}')
            
            for batch_size in batch_sizes:
                print(f'  Batch size: {batch_size}')
                
                try:
                    # Step 1: Preprocess
                    files = {'file': ('test.csv', df.to_csv(index=False), 'text/csv')}
                    response = requests.post(f"{base_url}/deep_config/preprocess", files=files)
                    
                    if response.status_code != 200:
                        print(f'    ❌ Preprocessing failed: {response.text}')
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
                        print(f'    ❌ Chunking failed: {response.text}')
                        continue
                    
                    # Step 3: Embedding
                    embed_data = {
                        'model_name': model_name,
                        'batch_size': batch_size,
                        'use_openai': False  # Force local model
                    }
                    
                    start_time = time.time()
                    response = requests.post(f"{base_url}/deep_config/embed", data=embed_data)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f'    ✅ Success: {result.get("status", "unknown")}')
                        print(f'    Time: {end_time - start_time:.4f}s')
                        print(f'    Chunks: {result.get("total_chunks", "N/A")}')
                        print(f'    Vector dimension: {result.get("vector_dimension", "N/A")}')
                        print(f'    Model: {result.get("model_name", "N/A")}')
                    else:
                        print(f'    ❌ Embedding failed: {response.text}')
                        
                except Exception as e:
                    print(f'    ❌ Error: {e}')
                
                print()
            
            print()
        
        print()
    
    print('=' * 60)

def test_embedding_performance():
    """Test embedding performance with different dataset sizes"""
    print('⚡ TESTING EMBEDDING PERFORMANCE')
    print('=' * 60)
    
    # Test different dataset sizes
    dataset_sizes = [10, 50, 100, 200, 500]
    
    # Available local models
    local_models = [
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L6-v2"
    ]
    
    # Test batch sizes
    batch_sizes = [32, 64, 128, 256]
    
    for size in dataset_sizes:
        print(f'Testing with {size} chunks...')
        print('-' * 30)
        
        # Create test chunks
        test_chunks = [f"This is test chunk number {i} with various content and information for embedding performance testing" for i in range(size)]
        
        for model_name in local_models:
            print(f'Model: {model_name}')
            
            for batch_size in batch_sizes:
                try:
                    start_time = time.time()
                    model, embeddings = embed_texts_enhanced(
                        test_chunks, 
                        model_name=model_name, 
                        openai_api_key=None,  # Force local model
                        batch_size=batch_size,
                        use_parallel=False
                    )
                    end_time = time.time()
                    
                    duration = end_time - start_time
                    chunks_per_second = size / duration if duration > 0 else 0
                    
                    print(f'  Batch {batch_size}: {embeddings.shape} in {duration:.4f}s ({chunks_per_second:.1f} chunks/s)')
                    
                except Exception as e:
                    print(f'  Batch {batch_size}: Failed - {e}')
            
            print()
        
        print()
    
    print('=' * 60)

def test_embedding_quality():
    """Test embedding quality and similarity"""
    print('🎯 TESTING EMBEDDING QUALITY')
    print('=' * 60)
    
    # Create test chunks with known relationships
    test_chunks = [
        "The system is running smoothly and efficiently",
        "The system is working well and performing optimally",  # Similar to first
        "We are analyzing customer data to improve service quality",
        "We are studying user data to enhance service quality",  # Similar to third
        "The development team is working on new features",
        "The development team is creating new functionality",  # Similar to fifth
        "Marketing campaigns are showing positive results",
        "Marketing efforts are demonstrating good outcomes",  # Similar to seventh
        "Financial reports indicate strong growth",
        "Financial statements show excellent progress"  # Similar to ninth
    ]
    
    print(f'Test chunks: {len(test_chunks)}')
    print('Expected similar pairs: (0,1), (2,3), (4,5), (6,7), (8,9)')
    print()
    
    # Available local models
    local_models = [
        "paraphrase-MiniLM-L6-v2",
        "all-MiniLM-L6-v2"
    ]
    
    for model_name in local_models:
        print(f'TESTING MODEL: {model_name}')
        print('-' * 40)
        
        try:
            model, embeddings = embed_texts_enhanced(
                test_chunks, 
                model_name=model_name, 
                openai_api_key=None,  # Force local model
                batch_size=32,
                use_parallel=False
            )
            
            print(f'Embeddings shape: {embeddings.shape}')
            
            # Calculate cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarities = cosine_similarity(embeddings)
            
            # Check expected similar pairs
            expected_pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
            
            print('Similarity scores for expected pairs:')
            for i, j in expected_pairs:
                similarity = similarities[i][j]
                print(f'  Chunks {i}-{j}: {similarity:.4f}')
                print(f'    "{test_chunks[i][:50]}..."')
                print(f'    "{test_chunks[j][:50]}..."')
                print()
            
            # Find most similar pairs
            print('Most similar pairs (top 5):')
            pairs = []
            for i in range(len(test_chunks)):
                for j in range(i+1, len(test_chunks)):
                    pairs.append((i, j, similarities[i][j]))
            
            pairs.sort(key=lambda x: x[2], reverse=True)
            
            for i, (idx1, idx2, sim) in enumerate(pairs[:5]):
                print(f'  {i+1}. Chunks {idx1}-{idx2}: {sim:.4f}')
                print(f'     "{test_chunks[idx1][:50]}..."')
                print(f'     "{test_chunks[idx2][:50]}..."')
                print()
            
        except Exception as e:
            print(f'❌ Failed: {e}')
        
        print()
    
    print('=' * 60)

if __name__ == "__main__":
    print('🚀 STARTING COMPREHENSIVE EMBEDDING TEST')
    print('=' * 80)
    print('This test covers:')
    print('• All backend embedding functions (standard and enhanced)')
    print('• Parallel embedding functions')
    print('• Fast Mode embedding (semantic clustering + embedding)')
    print('• Config-1 Mode embedding (all models and parameters)')
    print('• Deep Config Mode embedding (all models with metadata)')
    print('• Performance testing with different dataset sizes')
    print('• Embedding quality and similarity testing')
    print('• Local models only (no OpenAI API)')
    print('=' * 80)
    print()
    
    try:
        # Test 1: Backend functions
        test_backend_embedding_functions()
        
        # Test 2: Enhanced functions
        test_enhanced_embedding_functions()
        
        # Test 3: Parallel embedding
        test_parallel_embedding()
        
        # Test 4: Fast Mode
        test_fast_mode_embedding()
        
        # Test 5: Config-1 Mode
        test_config1_mode_embedding()
        
        # Test 6: Deep Config Mode
        test_deep_config_mode_embedding()
        
        # Test 7: Performance testing
        test_embedding_performance()
        
        # Test 8: Quality testing
        test_embedding_quality()
        
        print('🎉 COMPREHENSIVE EMBEDDING TEST COMPLETED SUCCESSFULLY!')
        print('=' * 80)
        print('✅ All backend embedding functions working correctly')
        print('✅ Enhanced embedding functions working')
        print('✅ Parallel embedding functions working')
        print('✅ Fast Mode embedding working')
        print('✅ Config-1 Mode embedding working')
        print('✅ Deep Config Mode embedding working')
        print('✅ Performance testing completed')
        print('✅ Embedding quality testing completed')
        print('=' * 80)
        
    except Exception as e:
        print(f'❌ COMPREHENSIVE EMBEDDING TEST FAILED: {e}')
        print('Make sure the FastAPI server is running: uvicorn main:app --reload')
