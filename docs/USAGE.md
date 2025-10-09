# Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Processing Modes](#processing-modes)
3. [File Upload](#file-upload)
4. [Database Integration](#database-integration)
5. [Configuration](#configuration)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### 1. Start the Application

```bash
# Start the API backend
python run_api.py

# Start the UI frontend (in another terminal)
python run_ui.py
```

### 2. Access the Interface

- **Frontend UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Basic Workflow

1. **Select Processing Mode**: Choose between Fast, Config-1, or Deep Config
2. **Upload Data**: Upload CSV file or connect to database
3. **Configure Parameters**: Set processing parameters based on mode
4. **Process Data**: Run the processing pipeline
5. **Search & Export**: Perform semantic search and export results

## Processing Modes

### Fast Mode ⚡

**Best for**: Quick processing with good results

**Features**:
- Automatic preprocessing
- Semantic clustering chunking
- Default paraphrase-MiniLM-L6-v2 model
- FAISS storage for fast retrieval
- Turbo mode for large files

**Configuration**:
- **Use OpenAI API**: Enable OpenAI embeddings (requires API key)
- **Turbo Mode**: Faster processing for large files
- **Batch Size**: Embedding generation batch size (64-512)

**Example**:
```python
# API call
POST /api/v1/process/fast
{
  "use_openai": false,
  "use_turbo": true,
  "batch_size": 256
}
```

### Config-1 Mode ⚙️

**Best for**: Balanced control and performance

**Features**:
- 4 chunking methods: Fixed, Recursive, Semantic, Document
- Multiple embedding models
- FAISS or ChromaDB storage
- Configurable retrieval metrics
- Advanced preprocessing options

**Configuration**:
- **Chunking Method**: Fixed, Recursive, Semantic, Document
- **Chunk Size**: Maximum characters per chunk (100-1000)
- **Overlap**: Character overlap between chunks (0-200)
- **Model Choice**: paraphrase-MiniLM-L6-v2, all-MiniLM-L6-v2, text-embedding-ada-002
- **Storage Choice**: FAISS, ChromaDB
- **Retrieval Metric**: Cosine, Dot Product, Euclidean

**Example**:
```python
# API call
POST /api/v1/process/config1
{
  "chunk_method": "recursive",
  "chunk_size": 400,
  "overlap": 50,
  "model_choice": "paraphrase-MiniLM-L6-v2",
  "storage_choice": "faiss",
  "retrieval_metric": "cosine"
}
```

### Deep Config Mode 🔬

**Best for**: Maximum control and quality

**Features**:
- 9-step workflow
- Advanced preprocessing
- Custom chunking configuration
- Multiple embedding options
- Persistent storage with metadata

**Workflow Steps**:
1. **Data Upload**: Upload CSV or connect to database
2. **Preprocessing**: Clean data, remove duplicates, normalize text
3. **Type Conversion**: Convert column data types
4. **Null Handling**: Handle missing values
5. **Text Processing**: Remove stopwords, lemmatize, stem
6. **Chunking**: Apply chosen chunking method
7. **Embedding**: Generate embeddings with selected model
8. **Storage**: Store in chosen vector database
9. **Complete**: Finalize processing and export results

**Example**:
```python
# API call
POST /api/v1/process/deep_config
{
  "preprocessing_config": "{\"remove_duplicates\": true, \"clean_headers\": true}",
  "chunking_config": "{\"method\": \"semantic\", \"chunk_size\": 400, \"overlap\": 50}",
  "embedding_config": "{\"model_name\": \"paraphrase-MiniLM-L6-v2\", \"batch_size\": 64}",
  "storage_config": "{\"type\": \"chroma\", \"collection_name\": \"my_collection\"}"
}
```

## File Upload

### Supported Formats

- **CSV files**: Comma-separated values
- **Encoding**: UTF-8 recommended
- **Size**: Up to 3GB+ (with large file support)

### Upload Process

1. **Select File**: Choose CSV file from your computer
2. **Preview Data**: Review data structure and content
3. **Configure Processing**: Set processing parameters
4. **Process**: Run the processing pipeline

### File Requirements

- **Column Headers**: Clear, descriptive column names
- **Data Types**: Consistent data types per column
- **Encoding**: UTF-8 encoding recommended
- **Size**: Consider file size for processing time

### Large File Handling

For files > 100MB:
- Enable turbo mode
- Increase batch size
- Consider database import
- Monitor system memory

## Database Integration

### Supported Databases

- **MySQL**: 5.7+
- **PostgreSQL**: 10+
- **SQLite**: Local files

### Connection Configuration

**MySQL**:
```json
{
  "db_type": "mysql",
  "host": "localhost",
  "port": 3306,
  "username": "your_username",
  "password": "your_password",
  "database": "your_database"
}
```

**PostgreSQL**:
```json
{
  "db_type": "postgresql",
  "host": "localhost",
  "port": 5432,
  "username": "your_username",
  "password": "your_password",
  "database": "your_database"
}
```

### Database Workflow

1. **Test Connection**: Verify database connectivity
2. **List Tables**: Browse available tables
3. **Select Table**: Choose table to import
4. **Preview Data**: Review table structure
5. **Import**: Import table data
6. **Process**: Run processing pipeline

### Large Table Handling

For tables > 100MB:
- Use chunked import
- Enable turbo mode
- Monitor memory usage
- Consider table views for filtered data

## Configuration

### Environment Variables

Create `.env` file:
```env
# API Configuration
API_HOST=localhost
API_PORT=8000
UI_PORT=8501

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Processing Configuration
LARGE_FILE_THRESHOLD=10485760
MAX_MEMORY_USAGE=0.8
BATCH_SIZE=2000
EMBEDDING_BATCH_SIZE=256
PARALLEL_WORKERS=6

# Storage Configuration
CHROMADB_PATH=storage/chromadb
FAISS_PATH=storage/faiss
CACHE_PATH=storage/cache
```

### Processing Parameters

#### Chunking Parameters
- **Chunk Size**: 100-1000 characters (default: 400)
- **Overlap**: 0-200 characters (default: 50)
- **Token Limit**: 500-5000 tokens (default: 2000)

#### Embedding Parameters
- **Batch Size**: 32-512 (default: 256)
- **Model Choice**: Local models or OpenAI API
- **Parallel Processing**: Enable for faster processing

#### Storage Parameters
- **Storage Type**: FAISS (fast) or ChromaDB (persistent)
- **Retrieval Metric**: Cosine, Dot Product, or Euclidean
- **Collection Name**: Custom name for ChromaDB collections

## Advanced Usage

### Custom Preprocessing

```python
from src.core.preprocessing import preprocess_auto_fast, clean_text_advanced

# Custom preprocessing
df = preprocess_auto_fast(df)

# Advanced text cleaning
df['text_column'] = clean_text_advanced(
    df['text_column'],
    lowercase=True,
    remove_delimiters=True,
    remove_whitespace=True
)
```

### Custom Chunking

```python
from src.core.chunking import chunk_semantic_cluster, document_based_chunking

# Semantic clustering
chunks = chunk_semantic_cluster(df, n_clusters=10)

# Document-based chunking
chunks, metadata = document_based_chunking(
    df,
    key_column='category',
    token_limit=2000,
    preserve_headers=True
)
```

### Custom Embedding

```python
from src.core.embedding import embed_texts

# Local model
model, embeddings = embed_texts(
    chunks,
    model_name="paraphrase-MiniLM-L6-v2",
    use_parallel=True
)

# OpenAI API
model, embeddings = embed_texts(
    chunks,
    model_name="text-embedding-ada-002",
    openai_api_key="your_api_key",
    openai_base_url="https://api.openai.com/v1"
)
```

### Custom Storage

```python
from src.core.storage import store_faiss, store_chroma

# FAISS storage
store_info = store_faiss(embeddings)

# ChromaDB storage
store_info = store_chroma(chunks, embeddings, "my_collection")
```

### Semantic Search

```python
from src.core.retrieval import retrieve_similar

# Basic search
results = retrieve_similar(
    query="machine learning algorithms",
    k=5,
    current_model=model,
    current_store_info=store_info,
    current_chunks=chunks,
    current_embeddings=embeddings
)

# Advanced search with filtering
results = retrieve_similar(
    query="machine learning algorithms",
    k=5,
    current_model=model,
    current_store_info=store_info,
    current_chunks=chunks,
    current_embeddings=embeddings,
    metadata_filter={"category": "AI", "year": "2023"}
)
```

## Best Practices

### Data Preparation

1. **Clean Data**: Remove duplicates, handle missing values
2. **Consistent Format**: Ensure consistent data types
3. **Clear Headers**: Use descriptive column names
4. **Encoding**: Use UTF-8 encoding
5. **Size**: Consider file size for processing time

### Chunking Strategy

1. **Fixed Chunking**: Good for uniform text length
2. **Recursive Chunking**: Good for variable text length
3. **Semantic Chunking**: Good for topic-based grouping
4. **Document Chunking**: Good for structured data

### Model Selection

1. **Local Models**: Faster, no API costs
2. **OpenAI API**: Higher quality, requires API key
3. **Batch Size**: Larger batches for faster processing
4. **Parallel Processing**: Enable for large datasets

### Storage Selection

1. **FAISS**: Fast similarity search, temporary storage
2. **ChromaDB**: Persistent storage, metadata support
3. **Retrieval Metric**: Cosine for general use, Dot for specific cases

### Performance Optimization

1. **Turbo Mode**: Enable for large files
2. **Batch Processing**: Use appropriate batch sizes
3. **Parallel Processing**: Enable for faster embedding
4. **Memory Management**: Monitor system memory
5. **Database Import**: Use for very large datasets

## Troubleshooting

### Common Issues

#### 1. Memory Issues
**Problem**: Out of memory errors
**Solutions**:
- Reduce batch size
- Enable turbo mode
- Use database import
- Increase system RAM

#### 2. Slow Processing
**Problem**: Processing takes too long
**Solutions**:
- Enable turbo mode
- Increase batch size
- Use parallel processing
- Consider OpenAI API

#### 3. Import Errors
**Problem**: Missing dependencies
**Solutions**:
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('popular')"
python -m spacy download en_core_web_sm
```

#### 4. API Connection Issues
**Problem**: Frontend can't connect to API
**Solutions**:
- Check API is running
- Verify firewall settings
- Check API_BASE_URL configuration

#### 5. Database Connection Issues
**Problem**: Can't connect to database
**Solutions**:
- Verify credentials
- Check network connectivity
- Ensure database server is running
- Check firewall settings

### Performance Tips

#### For Large Files (>100MB)
1. Use turbo mode
2. Increase batch size to 512
3. Use database import
4. Enable parallel processing
5. Consider OpenAI API

#### For High-Quality Results
1. Use Deep Config mode
2. Apply advanced preprocessing
3. Use semantic chunking
4. Use ChromaDB storage
5. Fine-tune parameters

#### For Production Use
1. Use Config-1 mode
2. Set up proper logging
3. Use persistent storage
4. Implement error handling
5. Set up monitoring

### Debugging

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check System Resources
```python
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
print(f"CPU: {psutil.cpu_percent()}%")
```

#### Monitor Processing
```python
import time
start_time = time.time()
# Processing code
print(f"Processing time: {time.time() - start_time:.2f}s")
```

### Getting Help

1. **Check Documentation**: Review API and usage docs
2. **Check Logs**: Review error logs for details
3. **Test Connectivity**: Verify API and database connections
4. **Check Resources**: Monitor system memory and CPU
5. **Contact Support**: Reach out for assistance

## Examples

### Example 1: Basic Text Processing

```python
import pandas as pd
from src.core.pipelines import run_fast_pipeline

# Load data
df = pd.read_csv('data.csv')

# Process with fast mode
result = run_fast_pipeline(
    df=df,
    file_info={'filename': 'data.csv', 'size': 1000},
    use_openai=False,
    use_turbo=True,
    batch_size=256
)

print(f"Processed {result['rows']} rows into {result['chunks']} chunks")
```

### Example 2: Custom Configuration

```python
from src.core.pipelines import run_config1_pipeline

# Process with custom configuration
result = run_config1_pipeline(
    df=df,
    chunk_method="semantic",
    chunk_size=500,
    overlap=100,
    model_choice="all-MiniLM-L6-v2",
    storage_choice="chroma",
    file_info={'filename': 'data.csv', 'size': 1000},
    use_openai=False,
    use_turbo=False,
    batch_size=128,
    retrieval_metric="cosine"
)

print(f"Processed {result['rows']} rows into {result['chunks']} chunks")
```

### Example 3: Database Integration

```python
from src.core.database import connect_mysql, import_table_to_dataframe
from src.core.pipelines import run_fast_pipeline

# Connect to database
conn = connect_mysql(
    host="localhost",
    port=3306,
    username="user",
    password="password",
    database="mydb"
)

# Import table
df = import_table_to_dataframe(conn, "my_table")

# Process data
result = run_fast_pipeline(
    df=df,
    file_info={'filename': 'my_table', 'file_type': 'database_table'},
    use_openai=False,
    use_turbo=True,
    batch_size=256
)

print(f"Processed {result['rows']} rows from database")
```

### Example 4: API Usage

```python
import requests

# Process file
with open('data.csv', 'rb') as f:
    files = {'file': ('data.csv', f, 'text/csv')}
    data = {'use_turbo': True, 'batch_size': 256}
    response = requests.post('http://localhost:8000/api/v1/process/fast', files=files, data=data)
    result = response.json()

# Search for similar content
search_data = {'query': 'machine learning', 'k': 5}
response = requests.post('http://localhost:8000/api/v1/retrieve', data=search_data)
results = response.json()

print(f"Found {len(results['results'])} similar chunks")
```
