# Chunk Optimizer v2.0 - Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Architecture](#architecture)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)

## 🎯 Overview

Chunk Optimizer is an advanced text chunking and embedding system that provides multiple processing modes for different use cases. It supports both local and OpenAI embedding models, various chunking algorithms, and multiple vector storage backends.

### Key Features

- **Multiple Processing Modes**: Fast, Config-1, and Deep Config modes
- **Advanced Chunking**: Fixed, Recursive, Semantic, and Document-based chunking
- **Multiple Embedding Models**: Local models and OpenAI API support
- **Vector Storage**: FAISS and ChromaDB support
- **Large File Support**: Handles files up to 3GB+
- **Database Integration**: MySQL and PostgreSQL support
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Modern UI**: Streamlit-based frontend with dark theme
- **Comprehensive Testing**: Unit and integration tests

## 🚀 Installation

### Prerequisites

- Python 3.8+
- 8GB+ RAM recommended
- 2GB+ disk space

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/chunkoptimizer/chunking-batch-processing.git
cd chunking-batch-processing

# Run setup script
python scripts/setup.py

# Install dependencies
pip install -r requirements.txt
```

### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('popular')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## 🏃 Quick Start

### 1. Start the Backend API

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

### 2. Start the Frontend UI

```bash
python run_ui.py
```

The UI will be available at `http://localhost:8501`

### 3. Access the Application

- **Frontend**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🏗️ Architecture

### Project Structure

```
chunking-batch-processing/
├── src/                           # Source code
│   ├── api/                       # FastAPI backend
│   │   ├── main.py               # FastAPI app
│   │   ├── routes/               # API routes
│   │   └── middleware.py         # Custom middleware
│   ├── core/                     # Core business logic
│   │   ├── pipelines.py         # Processing pipelines
│   │   ├── preprocessing.py      # Data preprocessing
│   │   ├── chunking.py          # Text chunking
│   │   ├── embedding.py          # Embedding generation
│   │   ├── storage.py           # Vector storage
│   │   ├── retrieval.py         # Semantic search
│   │   └── database.py          # Database operations
│   ├── ui/                       # Streamlit frontend
│   │   ├── app.py               # Main UI app
│   │   ├── components/          # UI components
│   │   └── utils/               # UI utilities
│   ├── models/                  # Data models
│   ├── utils/                   # Utility functions
│   └── config/                  # Configuration
├── tests/                       # Test suite
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
├── storage/                     # Persistent storage
├── data/                        # Data directory
└── requirements.txt             # Dependencies
```

### Processing Modes

#### 1. Fast Mode ⚡
- **Purpose**: Quick processing with good results
- **Features**: Auto-optimized pipeline, semantic clustering
- **Best For**: Large datasets, quick results
- **Processing Time**: 2-5 minutes for 10MB files

#### 2. Config-1 Mode ⚙️
- **Purpose**: Balanced control and performance
- **Features**: 4 chunking methods, multiple models, configurable storage
- **Best For**: Production environments, specific requirements
- **Processing Time**: 5-15 minutes for 10MB files

#### 3. Deep Config Mode 🔬
- **Purpose**: Maximum control and quality
- **Features**: 9-step workflow, advanced preprocessing, custom configuration
- **Best For**: Research, fine-tuning, complex requirements
- **Processing Time**: 15-30 minutes for 10MB files

## 📚 API Reference

### Core Endpoints

#### Processing Endpoints

**POST /api/v1/process/fast**
- Fast mode processing
- Parameters: file, use_openai, use_turbo, batch_size

**POST /api/v1/process/config1**
- Config-1 mode processing
- Parameters: file, chunk_method, chunk_size, model_choice, storage_choice

**POST /api/v1/process/deep_config**
- Deep config mode processing
- Parameters: file, preprocessing_config, chunking_config, embedding_config, storage_config

#### Database Endpoints

**POST /api/v1/db/test_connection**
- Test database connection
- Parameters: db_type, host, port, username, password, database

**POST /api/v1/db/list_tables**
- List database tables
- Parameters: db_type, host, port, username, password, database

**POST /api/v1/db/import_one**
- Import single table
- Parameters: db_type, host, port, username, password, database, table_name

#### Retrieval Endpoints

**POST /api/v1/retrieve**
- Semantic search
- Parameters: query, k, metadata_filter

**POST /api/v1/retrieve/advanced**
- Advanced semantic search
- Parameters: query, k, metadata_filter, similarity_threshold

#### Export Endpoints

**GET /api/v1/export/chunks**
- Export chunks as CSV

**GET /api/v1/export/embeddings**
- Export embeddings as numpy array

**GET /api/v1/export/embeddings_text**
- Export embeddings as JSON

#### System Endpoints

**GET /api/v1/health**
- Health check

**GET /api/v1/system_info**
- System information

**GET /api/v1/capabilities**
- System capabilities

### Request/Response Examples

#### Fast Mode Processing

```python
import requests

# Upload file and process
with open('data.csv', 'rb') as f:
    files = {'file': ('data.csv', f, 'text/csv')}
    data = {
        'use_openai': False,
        'use_turbo': True,
        'batch_size': 256
    }
    response = requests.post('http://localhost:8000/api/v1/process/fast', files=files, data=data)
    result = response.json()

print(f"Processed {result['rows']} rows into {result['chunks']} chunks")
print(f"Storage: {result['stored']}, Model: {result['embedding_model']}")
```

#### Semantic Search

```python
import requests

# Search for similar content
data = {
    'query': 'machine learning algorithms',
    'k': 5
}
response = requests.post('http://localhost:8000/api/v1/retrieve', data=data)
results = response.json()

for result in results['results']:
    print(f"Similarity: {result['similarity']:.3f}")
    print(f"Content: {result['content'][:100]}...")
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_HOST=localhost
API_PORT=8000
UI_PORT=8501

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Database Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USERNAME=your_username
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database

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

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Configuration Files

#### pyproject.toml
- Project metadata and dependencies
- Tool configurations (black, mypy, pytest)

#### requirements.txt
- Python package dependencies
- Version specifications

## 💡 Usage Examples

### Example 1: Basic Text Processing

```python
from src.core.pipelines import run_fast_pipeline
import pandas as pd

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

### Example 2: Custom Chunking Configuration

```python
from src.core.pipelines import run_config1_pipeline
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

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

### Example 3: Deep Config Processing

```python
from src.core.pipelines import run_deep_config_pipeline
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Configure deep processing
config_dict = {
    "preprocessing": {
        "remove_duplicates": True,
        "clean_headers": True,
        "normalize_text": True
    },
    "chunking": {
        "method": "document",
        "chunk_size": 400,
        "overlap": 50,
        "key_column": "category",
        "token_limit": 2000,
        "preserve_headers": True
    },
    "embedding": {
        "model_name": "paraphrase-MiniLM-L6-v2",
        "batch_size": 64,
        "use_parallel": True
    },
    "storage": {
        "type": "chroma",
        "collection_name": "my_collection"
    }
}

# Process with deep config
result = run_deep_config_pipeline(
    df=df,
    config_dict=config_dict,
    file_info={'filename': 'data.csv', 'size': 1000}
)

print(f"Processed {result['rows']} rows into {result['chunks']} chunks")
```

### Example 4: Database Integration

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

## 🔧 Troubleshooting

### Common Issues

#### 1. Memory Issues
**Problem**: Out of memory errors with large files
**Solution**: 
- Reduce batch size
- Enable turbo mode
- Use database import for very large files
- Increase system RAM

#### 2. Import Errors
**Problem**: Missing dependencies
**Solution**:
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('popular')"
python -m spacy download en_core_web_sm
```

#### 3. API Connection Issues
**Problem**: Frontend can't connect to API
**Solution**:
- Check API is running on correct port
- Verify firewall settings
- Check API_BASE_URL in frontend

#### 4. Database Connection Issues
**Problem**: Can't connect to database
**Solution**:
- Verify database credentials
- Check network connectivity
- Ensure database server is running
- Check firewall settings

#### 5. Slow Processing
**Problem**: Processing takes too long
**Solution**:
- Enable turbo mode
- Increase batch size
- Use parallel processing
- Consider using OpenAI API for faster embedding

### Performance Optimization

#### For Large Files (>100MB)
1. Use turbo mode
2. Increase batch size to 512
3. Use database import instead of file upload
4. Enable parallel processing
5. Consider using OpenAI API

#### For High-Quality Results
1. Use Deep Config mode
2. Apply advanced preprocessing
3. Use semantic chunking
4. Use ChromaDB for persistent storage
5. Fine-tune chunking parameters

#### For Production Use
1. Use Config-1 mode
2. Set up proper logging
3. Use persistent storage (ChromaDB)
4. Implement error handling
5. Set up monitoring

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Set up development environment:
   ```bash
   git clone https://github.com/yourusername/chunking-batch-processing.git
   cd chunking-batch-processing
   python scripts/setup.py
   pip install -r requirements.txt
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_preprocessing.py

# Run with coverage
pytest --cov=src tests/

# Run integration tests
pytest tests/integration/
```

### Code Style

- Follow PEP 8
- Use black for formatting
- Use mypy for type checking
- Write comprehensive tests
- Update documentation

### Submitting Changes

1. Write tests for new features
2. Update documentation
3. Run tests and linting
4. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAPI team for the excellent web framework
- Streamlit team for the amazing UI framework
- Hugging Face for the transformer models
- Facebook AI Research for FAISS
- ChromaDB team for the vector database
- OpenAI for the embedding API

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/chunkoptimizer/chunking-batch-processing/wiki)
- **Issues**: [GitHub Issues](https://github.com/chunkoptimizer/chunking-batch-processing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chunkoptimizer/chunking-batch-processing/discussions)
- **Email**: support@chunkoptimizer.com

---

**Chunk Optimizer v2.0** - Advanced Text Processing and Embedding System
