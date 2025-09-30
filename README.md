# 📦 Chunking Optimizer

A powerful text processing and semantic search application that converts CSV files and database tables into searchable vector embeddings.

## 🚀 Features

- **Multiple Data Sources**: Import from CSV files or database tables (MySQL/PostgreSQL)
- **Flexible Processing**: Three processing modes (Fast, Config-1, Deep)
- **Smart Chunking**: Fixed, recursive, semantic, and document-based chunking strategies
- **Vector Search**: Semantic similarity search using FAISS or ChromaDB
- **Export Options**: Download processed chunks and embeddings
- **Real-time Monitoring**: Process tracking and system information

## 🏗️ Architecture

```
Frontend (Streamlit) ←→ Backend (FastAPI) ←→ Processing Pipeline
     ↓                        ↓                      ↓
   User Interface         API Endpoints        Text Processing
   - File Upload         - CSV Processing      - Chunking
   - DB Connection       - DB Import           - Embeddings
   - Search Interface    - Vector Storage      - Vector DB
   - Results Display     - Export Functions    - Retrieval
```

## 📋 Prerequisites

- Python 3.8+
- MySQL (optional, for database import)
- PostgreSQL (optional, for database import)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd streamlit_29-09
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application**
   ```bash
   # Terminal 1 - Backend API
   uvicorn main:app --reload --port 8000
   
   # Terminal 2 - Frontend UI
   streamlit run app.py
   ```

4. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 📊 Usage

### CSV File Processing

1. **Upload CSV File**
   - Click "📄 CSV File Upload"
   - Select your CSV file
   - Preview the data (optional)

2. **Choose Processing Mode**
   - **Fast Mode**: Auto-optimized pipeline with semantic clustering
   - **Config-1 Mode**: Customizable preprocessing and chunking
   - **Deep Mode**: Advanced text processing with NLP features

3. **Run Pipeline**
   - Click "🚀 Run Pipeline"
   - Monitor progress in the sidebar
   - View results and statistics

### Database Import

1. **Connect to Database**
   - Click "🗄️ Database Connection"
   - Select database type (MySQL/PostgreSQL)
   - Enter connection details:
     - Host: localhost
     - Port: 3306 (MySQL) or 5432 (PostgreSQL)
     - Username: your_db_username
     - Password: your_db_password
     - Database: your_database_name

2. **Test Connection**
   - Click "🔌 Test" to verify connection
   - Click "📋 Tables" to list available tables

3. **Import and Process**
   - Select a table from the dropdown
   - Click "📥 Import One (Fast Mode)"
   - Monitor processing in the sidebar

### Semantic Search

After processing, use the semantic search feature:

1. **Enter Search Query**
   - Type your question in natural language
   - Adjust "Top K results" slider (1-10)

2. **View Results**
   - Results ranked by similarity score
   - Color-coded relevance (green > 0.7, yellow > 0.4, red < 0.4)

### Export Results

- **Download Chunks**: Export processed text chunks as TXT file
- **Download Embeddings**: Export vector embeddings as NPY file

## ⚙️ Configuration

### Processing Modes

#### Fast Mode
- **Preprocessing**: Auto drop nulls
- **Chunking**: Semantic clustering
- **Embeddings**: MiniLM-L6-v2
- **Storage**: FAISS
- **Best for**: Quick processing, good results

#### Config-1 Mode
- **Preprocessing**: Customizable null handling
- **Chunking**: Fixed, recursive, semantic, or document-based
- **Embeddings**: MiniLM models
- **Storage**: FAISS or ChromaDB
- **Best for**: Balanced control and performance

#### Deep Mode
- **Preprocessing**: Advanced NLP (stopwords, stemming, lemmatization)
- **Chunking**: All chunking methods available
- **Embeddings**: Multiple model options
- **Storage**: FAISS or ChromaDB
- **Best for**: Maximum control and quality

### Chunking Strategies

- **Fixed**: Split text into fixed-size chunks with overlap
- **Recursive**: Smart splitting that respects word boundaries
- **Semantic**: Group similar content using embeddings
- **Document**: Group by key columns without overlap

## 🗄️ Database Setup

### MySQL Setup

1. **Install MySQL**
   - Download from https://dev.mysql.com/downloads/mysql/
   - Install with default settings
   - Remember root password

2. **Create Test Database**
   ```sql
   CREATE DATABASE csv_chunker;
   USE csv_chunker;
   
   CREATE TABLE customers (
     id INT AUTO_INCREMENT PRIMARY KEY,
     name VARCHAR(100) NOT NULL,
     city VARCHAR(100),
     payment_method VARCHAR(50),
     amount DECIMAL(10,2)
   );
   
   INSERT INTO customers VALUES
   (1, 'John', 'Chennai', 'UPI', 1200.50),
   (2, 'Jane', 'Mumbai', 'Card', 899.99);
   ```

### PostgreSQL Setup

1. **Install PostgreSQL**
   - Download from https://www.postgresql.org/download/
   - Install with default settings
   - Remember postgres password

2. **Create Test Database**
   ```sql
   CREATE DATABASE csv_chunker;
   \c csv_chunker
   
   CREATE TABLE customers (
     id SERIAL PRIMARY KEY,
     name VARCHAR(100) NOT NULL,
     city VARCHAR(100),
     payment_method VARCHAR(50),
     amount NUMERIC(10,2)
   );
   
   INSERT INTO customers VALUES
   (1, 'John', 'Chennai', 'UPI', 1200.50),
   (2, 'Jane', 'Mumbai', 'Card', 899.99);
   ```

## 📁 Project Structure

```
streamlit_29-09/
├── app.py                 # Streamlit frontend
├── main.py               # FastAPI backend
├── backend.py            # Core processing logic
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── chunking_data.db     # SQLite database (auto-created)
```

## 🔧 API Endpoints

### CSV Processing
- `POST /run_fast` - Fast mode processing
- `POST /run_config1` - Config-1 mode processing
- `POST /run_deep` - Deep mode processing

### Database Import
- `POST /db/test_connection` - Test database connection
- `POST /db/list_tables` - Get table list
- `POST /db/import_one` - Import single table

### Search & Export
- `POST /retrieve` - Semantic search
- `GET /export/chunks` - Download chunks
- `GET /export/embeddings` - Download embeddings

### System Info
- `GET /system_info` - System memory usage
- `GET /file_info` - File information
- `GET /health` - Health check

## 🚨 Troubleshooting

### Common Issues

1. **Import Error: mysql.connector**
   ```bash
   pip install mysql-connector-python
   ```

2. **Import Error: psycopg2**
   ```bash
   pip install psycopg2-binary
   ```

3. **Connection Refused**
   - Check if MySQL/PostgreSQL service is running
   - Verify host, port, username, password
   - Ensure database exists

4. **Memory Issues with Large Files**
   - Use smaller files (< 100MB)
   - Increase system RAM
   - Use batch processing for very large datasets

5. **Slow Processing**
   - Use Fast mode for quick results
   - Reduce chunk size for faster processing
   - Use FAISS instead of ChromaDB for better performance

### Performance Tips

- **Small files (< 10MB)**: Any mode works well
- **Medium files (10-100MB)**: Fast or Config-1 mode
- **Large files (> 100MB)**: Consider batch processing
- **Memory**: Ensure 2-4GB free RAM for processing

## 📈 File Size Limits

- **Default**: 200MB per file
- **Recommended**: 10-50MB for optimal performance
- **Maximum**: 500MB (with configuration changes)
- **Memory usage**: ~3-4x file size during processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation at http://localhost:8000/docs
3. Create an issue in the repository

---

