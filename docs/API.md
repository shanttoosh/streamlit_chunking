# API Documentation

## Overview

The Chunk Optimizer API provides RESTful endpoints for text processing, chunking, embedding generation, and semantic search. The API is built with FastAPI and provides automatic OpenAPI documentation.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing API key authentication or OAuth2.

## Content Types

- **Request**: `application/json` or `multipart/form-data` (for file uploads)
- **Response**: `application/json`

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error message description"
}
```

HTTP status codes:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

## Endpoints

### Health Check

#### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running"
}
```

#### GET /api/v1/health

Detailed health check with system information.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running"
}
```

### System Information

#### GET /api/v1/system_info

Get system information including memory usage and capabilities.

**Response:**
```json
{
  "memory_usage": "45.2%",
  "available_memory": "8.5 GB",
  "total_memory": "16.0 GB",
  "large_file_support": true,
  "max_recommended_file_size": "3GB+",
  "embedding_batch_size": 256,
  "parallel_workers": 6
}
```

#### GET /api/v1/capabilities

Get system capabilities and supported features.

**Response:**
```json
{
  "large_file_support": true,
  "performance_features": {
    "turbo_mode": true,
    "parallel_processing": true,
    "batch_processing": true
  },
  "supported_formats": ["csv"],
  "supported_databases": ["mysql", "postgresql", "sqlite"],
  "supported_models": [
    "paraphrase-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
    "text-embedding-ada-002"
  ],
  "supported_storage": ["faiss", "chroma"],
  "supported_chunking": ["fixed", "recursive", "semantic", "document"]
}
```

### Processing Endpoints

#### POST /api/v1/process/fast

Process data using Fast Mode with automatic optimization.

**Parameters:**
- `file` (file, optional): CSV file to process
- `use_openai` (boolean): Use OpenAI API for embeddings
- `openai_api_key` (string, optional): OpenAI API key
- `openai_base_url` (string, optional): OpenAI API base URL
- `use_turbo` (boolean): Enable turbo mode for faster processing
- `batch_size` (integer): Batch size for embedding generation

**Response:**
```json
{
  "rows": 1000,
  "chunks": 250,
  "stored": "faiss",
  "embedding_model": "paraphrase-MiniLM-L6-v2",
  "retrieval_ready": true,
  "turbo_mode": true
}
```

#### POST /api/v1/process/config1

Process data using Config-1 Mode with customizable parameters.

**Parameters:**
- `file` (file, optional): CSV file to process
- `chunk_method` (string): Chunking method ("fixed", "recursive", "semantic", "document")
- `chunk_size` (integer): Maximum characters per chunk
- `overlap` (integer): Character overlap between chunks
- `model_choice` (string): Embedding model to use
- `storage_choice` (string): Storage backend ("faiss", "chroma")
- `retrieval_metric` (string): Similarity metric ("cosine", "dot", "euclidean")
- `use_openai` (boolean): Use OpenAI API for embeddings
- `openai_api_key` (string, optional): OpenAI API key
- `openai_base_url` (string, optional): OpenAI API base URL
- `use_turbo` (boolean): Enable turbo mode
- `batch_size` (integer): Batch size for embedding generation

**Response:**
```json
{
  "rows": 1000,
  "chunks": 200,
  "stored": "chroma",
  "embedding_model": "all-MiniLM-L6-v2",
  "retrieval_ready": true,
  "turbo_mode": false
}
```

#### POST /api/v1/process/deep_config

Process data using Deep Config Mode with advanced configuration.

**Parameters:**
- `file` (file, optional): CSV file to process
- `preprocessing_config` (string): JSON string with preprocessing configuration
- `chunking_config` (string): JSON string with chunking configuration
- `embedding_config` (string): JSON string with embedding configuration
- `storage_config` (string): JSON string with storage configuration

**Example Configuration:**
```json
{
  "preprocessing_config": "{\"remove_duplicates\": true, \"clean_headers\": true}",
  "chunking_config": "{\"method\": \"semantic\", \"chunk_size\": 400, \"overlap\": 50}",
  "embedding_config": "{\"model_name\": \"paraphrase-MiniLM-L6-v2\", \"batch_size\": 64}",
  "storage_config": "{\"type\": \"chroma\", \"collection_name\": \"my_collection\"}"
}
```

**Response:**
```json
{
  "rows": 1000,
  "chunks": 180,
  "stored": "chroma",
  "embedding_model": "paraphrase-MiniLM-L6-v2",
  "retrieval_ready": true,
  "processing_time": 45.2,
  "enhanced_pipeline": true
}
```

### Database Endpoints

#### POST /api/v1/db/test_connection

Test database connection.

**Parameters:**
- `db_type` (string): Database type ("mysql", "postgresql")
- `host` (string): Database host
- `port` (integer): Database port
- `username` (string): Database username
- `password` (string): Database password
- `database` (string): Database name

**Response:**
```json
{
  "connected": true,
  "message": "Connection successful"
}
```

#### POST /api/v1/db/list_tables

List tables in the database.

**Parameters:**
- `db_type` (string): Database type ("mysql", "postgresql")
- `host` (string): Database host
- `port` (integer): Database port
- `username` (string): Database username
- `password` (string): Database password
- `database` (string): Database name

**Response:**
```json
{
  "success": true,
  "tables": ["table1", "table2", "table3"]
}
```

#### POST /api/v1/db/import_one

Import a single table from the database.

**Parameters:**
- `db_type` (string): Database type ("mysql", "postgresql")
- `host` (string): Database host
- `port` (integer): Database port
- `username` (string): Database username
- `password` (string): Database password
- `database` (string): Database name
- `table_name` (string): Table name to import

**Response:**
```json
{
  "success": true,
  "message": "Table imported successfully",
  "table_name": "my_table",
  "rows": 5000,
  "columns": 10,
  "column_names": ["id", "name", "description", "category", "score"]
}
```

### Retrieval Endpoints

#### POST /api/v1/retrieve

Perform semantic search on processed data.

**Parameters:**
- `query` (string): Search query
- `k` (integer): Number of results to return
- `metadata_filter` (string, optional): JSON string with metadata filters

**Response:**
```json
{
  "query": "machine learning algorithms",
  "k": 5,
  "results": [
    {
      "rank": 1,
      "content": "Machine learning algorithms are computational methods...",
      "similarity": 0.892,
      "distance": 0.108,
      "metadata": {
        "chunk_id": 42,
        "source": "ml_textbook.pdf"
      }
    }
  ]
}
```

#### POST /api/v1/retrieve/advanced

Perform advanced semantic search with filtering and thresholds.

**Parameters:**
- `query` (string): Search query
- `k` (integer): Number of results to return
- `metadata_filter` (string, optional): JSON string with metadata filters
- `similarity_threshold` (float, optional): Minimum similarity threshold

**Response:**
```json
{
  "query": "machine learning algorithms",
  "k": 5,
  "results": [
    {
      "rank": 1,
      "content": "Machine learning algorithms are computational methods...",
      "similarity": 0.892,
      "distance": 0.108,
      "metadata": {
        "chunk_id": 42,
        "source": "ml_textbook.pdf"
      }
    }
  ],
  "filtered_count": 3
}
```

#### GET /api/v1/retrieve/status

Get current retrieval system status.

**Response:**
```json
{
  "model_loaded": true,
  "store_loaded": true,
  "chunks_loaded": true,
  "embeddings_loaded": true,
  "total_chunks": 1000,
  "store_type": "faiss"
}
```

### Export Endpoints

#### GET /api/v1/export/chunks

Export processed chunks as CSV file.

**Response:**
- Content-Type: `text/csv`
- File: `chunks.csv`

#### GET /api/v1/export/embeddings

Export embeddings as numpy array file.

**Response:**
- Content-Type: `application/octet-stream`
- File: `embeddings.npy`

#### GET /api/v1/export/embeddings_text

Export embeddings as JSON file with text.

**Response:**
- Content-Type: `application/json`
- File: `embeddings.json`

**Example Response:**
```json
{
  "total_chunks": 1000,
  "vector_dimension": 384,
  "embeddings": [
    {
      "chunk_id": 1,
      "chunk_text": "This is a sample chunk...",
      "embedding": [0.1, 0.2, 0.3, ...]
    }
  ]
}
```

#### GET /api/v1/export/preprocessed

Export preprocessed data as CSV file.

**Response:**
- Content-Type: `text/csv`
- File: `preprocessed_data.csv`

#### GET /api/v1/export/deep_chunks

Export deep config chunks with metadata.

**Response:**
- Content-Type: `text/csv`
- File: `deep_chunks.csv`

#### GET /api/v1/export/deep_embeddings

Export deep config embeddings with metadata.

**Response:**
- Content-Type: `application/json`
- File: `deep_embeddings.json`

#### GET /api/v1/export/status

Get current export status.

**Response:**
```json
{
  "chunks_available": true,
  "embeddings_available": true,
  "preprocessed_data_available": true,
  "total_chunks": 1000,
  "total_embeddings": 1000
}
```

## Rate Limiting

Currently, there are no rate limits implemented. For production deployments, consider implementing rate limiting to prevent abuse.

## CORS

CORS is enabled for all origins. For production deployments, restrict CORS to specific domains.

## WebSocket Support

WebSocket support is not currently implemented. Consider adding WebSocket endpoints for real-time processing updates.

## SDK Examples

### Python SDK

```python
import requests

class ChunkOptimizerClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def process_fast(self, file_path, **kwargs):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/api/v1/process/fast",
                files=files,
                data=kwargs
            )
        return response.json()
    
    def retrieve(self, query, k=5):
        data = {'query': query, 'k': k}
        response = requests.post(
            f"{self.base_url}/api/v1/retrieve",
            data=data
        )
        return response.json()
    
    def export_chunks(self):
        response = requests.get(f"{self.base_url}/api/v1/export/chunks")
        return response.content

# Usage
client = ChunkOptimizerClient()
result = client.process_fast('data.csv', use_turbo=True)
search_results = client.retrieve('machine learning', k=10)
```

### JavaScript SDK

```javascript
class ChunkOptimizerClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async processFast(file, options = {}) {
        const formData = new FormData();
        formData.append('file', file);
        
        Object.keys(options).forEach(key => {
            formData.append(key, options[key]);
        });
        
        const response = await fetch(`${this.baseUrl}/api/v1/process/fast`, {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    }
    
    async retrieve(query, k = 5) {
        const response = await fetch(`${this.baseUrl}/api/v1/retrieve`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query, k })
        });
        
        return await response.json();
    }
}

// Usage
const client = new ChunkOptimizerClient();
const fileInput = document.getElementById('fileInput');
const file = fileInput.files[0];

client.processFast(file, { use_turbo: true })
    .then(result => console.log(result))
    .catch(error => console.error(error));
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error - Server error |

## Changelog

### v2.0.0
- Added modular architecture
- Added comprehensive API endpoints
- Added database integration
- Added advanced chunking methods
- Added multiple storage backends
- Added comprehensive testing

### v1.0.0
- Initial release
- Basic chunking functionality
- Single storage backend
- Basic API endpoints
