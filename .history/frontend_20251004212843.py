# app.py (Streamlit Frontend) - COMPLETE UPDATED VERSION
import streamlit as st
import pandas as pd
import requests
import io
import time
import base64
import os
from datetime import datetime
import json
import tempfile
import shutil

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"

# Helper functions for enhanced preprocessing
def validate_and_normalize_headers(columns):
    """
    Validate and normalize column headers
    """
    new_columns = []
    for i, col in enumerate(columns):
        if col is None or str(col).strip() == "":
            new_col = f"column_{i+1}"
        else:
            new_col = str(col).strip().lower()
        new_columns.append(new_col)
    return new_columns

def normalize_text_column(s: pd.Series, lowercase=True, strip=True, remove_html_flag=True):
    """
    Normalize text column with HTML removal, lowercase, and whitespace cleanup
    """
    s = s.fillna('')
    
    if remove_html_flag:
        # Simple HTML tag removal
        s = s.map(lambda x: re.sub(r'<[^<]+?>', ' ', str(x)) if isinstance(x, str) else x)
    
    if lowercase:
        s = s.map(lambda x: x.lower() if isinstance(x, str) else x)
    
    if strip:
        s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Replace multiple whitespace with single space
    s = s.map(lambda x: re.sub(r'\s+', ' ', str(x)) if isinstance(x, str) else x)
    
    return s

def remove_stopwords_from_text_columns(df: pd.DataFrame, remove_stopwords: bool = True):
    """
    Remove stopwords from text columns using spaCy
    """
    if not remove_stopwords:
        return df, "Stopwords removal skipped"
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except ImportError:
        st.warning("spaCy not available for stopwords removal")
        return df, "spaCy not available for stopwords removal"
    
    df_copy = df.copy()
    text_cols = df_copy.select_dtypes(include=["object"]).columns
    
    if text_cols.empty:
        return df, "No text columns found for stopwords removal"
    
    processed_cols = []
    
    def process_text(text):
        doc = nlp(str(text))
        filtered_tokens = [token.text for token in doc if not token.is_stop]
        return " ".join(filtered_tokens)
    
    for col in text_cols:
        df_copy[col] = df_copy[col].apply(process_text)
        processed_cols.append(col)
    
    message = f"Stopwords removed from columns: {', '.join(processed_cols)}"
    return df_copy, message

def apply_text_normalization(df: pd.DataFrame, method: str = "lemmatize"):
    """
    Apply text normalization (lemmatization or stemming) to text columns
    """
    if method not in ["lemmatize", "stem"]:
        return df, f"Invalid normalization method: {method}"
    
    df_copy = df.copy()
    text_cols = df_copy.select_dtypes(include=["object"]).columns
    
    if text_cols.empty:
        return df, "No text columns found for normalization"
    
    if method == "stem":
        try:
            from nltk.stem import PorterStemmer
            from nltk.tokenize import word_tokenize
            stemmer = PorterStemmer()
            
            def stem_text(text):
                words = word_tokenize(str(text))
                return " ".join([stemmer.stem(word) for word in words])
            
            for col in text_cols:
                df_copy[col] = df_copy[col].apply(stem_text)
                
        except ImportError:
            st.warning("NLTK not available for stemming")
            return df, "NLTK not available for stemming"
            
    elif method == "lemmatize":
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            
            def lemmatize_text(text):
                doc = nlp(str(text))
                return " ".join([token.text if token.lemma_ == '-PRON-' else token.lemma_ for token in doc])
            
            for col in text_cols:
                df_copy[col] = df_copy[col].apply(lemmatize_text)
                
        except ImportError:
            st.warning("spaCy not available for lemmatization")
            return df, "spaCy not available for lemmatization"
    
    message = f"Applied {method} normalization to {len(text_cols)} text columns"
    return df_copy, message

# ---------- Enhanced Orange-Grey Theme with Dark Mode Support ----------
st.markdown("""
<style>
    /* Enhanced Orange-Grey Theme with Dark Mode Support */
    :root {
        --primary: #FF8C00;
        --secondary: #FFA500;
        --accent: #FFB74D;
        --dark: #2C3E50;
        --medium: #34495E;
        --light: #ECF0F1;
        --text: #2C3E50;
        --success: #28a745;
        --warning: #ffc107;
        --danger: #dc3545;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        border-left: 4px solid var(--primary) !important;
        padding-left: 10px !important;
    }
    
    /* Cards with dark theme */
    .custom-card {
        background: #2d2d2d;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        border-left: 4px solid var(--primary);
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }
    
    .card-title {
        color: #ffffff;
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .card-content {
        color: #cccccc;
        font-size: 0.95em;
        line-height: 1.5;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, var(--secondary), var(--accent)) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4) !important;
    }
    
    /* Process steps */
    .process-step {
        background: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #666;
        transition: all 0.3s ease;
    }
    
    .process-step.running {
        border-left-color: var(--warning);
        background: linear-gradient(90deg, #2d2d2d, #444);
    }
    
    .process-step.completed {
        border-left-color: var(--success);
        background: linear-gradient(90deg, #2d2d2d, #2d4a2d);
    }
    
    .process-step.pending {
        border-left-color: #666;
        background: #2d2d2d;
    }
    
    /* Dataframes */
    .dataframe {
        background: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* Text inputs - WIDER */
    .stTextInput > div > div {
        width: 100% !important;
    }
    
    .stTextInput > div > div > input {
        background: #2d2d2d;
        color: #ffffff;
        border: 1px solid #555;
        width: 100% !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 1px var(--primary);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #2d2d2d;
        color: #ffffff;
        border: 1px solid #555;
    }
    
    /* Number inputs */
    .stNumberInput > div > div > input {
        background: #2d2d2d;
        color: #ffffff;
        border: 1px solid #555;
    }
    
    /* Checkboxes */
    .stCheckbox > label {
        color: #ffffff !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #ffffff !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--dark) 0%, var(--medium) 100%) !important;
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(45deg, #155724, #1e7e34) !important;
        color: #ffffff !important;
        border-left: 4px solid var(--success) !important;
    }
    
    /* Error messages */
    .stError {
        background: linear-gradient(45deg, #721c24, #c82333) !important;
        color: #ffffff !important;
        border-left: 4px solid var(--danger) !important;
    }
    
    /* Warning messages */
    .stWarning {
        background: linear-gradient(45deg, #856404, #e0a800) !important;
        color: #ffffff !important;
        border-left: 4px solid var(--warning) !important;
    }
    
    /* Info messages */
    .stInfo {
        background: linear-gradient(45deg, #0c5460, #138496) !important;
        color: #ffffff !important;
        border-left: 4px solid #17a2b8 !important;
    }
    
    /* Center align text inputs */
    div[data-testid="stTextInput"] {
        width: 100% !important;
    }
    
    /* Make text areas wider */
    .stTextArea > div > div > textarea {
        width: 100% !important;
    }
    
    /* Preview table styling */
    .preview-table {
        background: #2d2d2d;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Large file warning */
    .large-file-warning {
        background: linear-gradient(45deg, #856404, #e0a800);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid var(--warning);
    }
    
    /* File upload styling */
    .uploadedFile {
        background: #2d2d2d;
        border: 2px dashed #666;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# ---------- API Client Functions ----------
def call_fast_api(file_path: str, filename: str, db_type: str, db_config: dict = None, 
                  use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                  process_large_files: bool = True):
    """Send file directly from filesystem path"""
    try:
        with open(file_path, 'rb') as f:
            if db_config and db_config.get('use_db'):
                data = {
                    "db_type": db_config["db_type"],
                    "host": db_config["host"],
                    "port": db_config["port"],
                    "username": db_config["username"],
                    "password": db_config["password"],
                    "database": db_config["database"],
                    "table_name": db_config["table_name"],
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files
                }
                response = requests.post(f"{API_BASE_URL}/run_fast", data=data)
            else:
                files = {"file": (filename, f, "text/csv")}
                data = {
                    "db_type": db_type,
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files
                }
                response = requests.post(f"{API_BASE_URL}/run_fast", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def call_config1_api(file_path: str, filename: str, config: dict, db_config: dict = None,
                    use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                    process_large_files: bool = True):
    """Send file directly from filesystem path"""
    try:
        with open(file_path, 'rb') as f:
            if db_config and db_config.get('use_db'):
                data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
                data.update({
                    "db_type": db_config["db_type"],
                    "host": db_config["host"],
                    "port": db_config["port"],
                    "username": db_config["username"],
                    "password": db_config["password"],
                    "database": db_config["database"],
                    "table_name": db_config["table_name"],
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files
                })
                response = requests.post(f"{API_BASE_URL}/run_config1", data=data)
            else:
                files = {"file": (filename, f, "text/csv")}
                data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
                data.update({
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files
                })
                response = requests.post(f"{API_BASE_URL}/run_config1", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def call_deep_api(file_path: str, filename: str, config: dict, db_config: dict = None,
                 use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                 process_large_files: bool = True):
    """Send file directly from filesystem path"""
    try:
        with open(file_path, 'rb') as f:
            if db_config and db_config.get('use_db'):
                data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
                data.update({
                    "db_type": db_config["db_type"],
                    "host": db_config["host"],
                    "port": db_config["port"],
                    "username": db_config["username"],
                    "password": db_config["password"],
                    "database": db_config["database"],
                    "table_name": db_config["table_name"],
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files
                })
                response = requests.post(f"{API_BASE_URL}/run_deep", data=data)
            else:
                files = {"file": (filename, f, "text/csv")}
                data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
                data.update({
                    "use_openai": use_openai,
                    "openai_api_key": openai_api_key,
                    "openai_base_url": openai_base_url,
                    "process_large_files": process_large_files
                })
                response = requests.post(f"{API_BASE_URL}/run_deep", files=files, data=data)
        return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def call_retrieve_api(query: str, k: int = 5):
    data = {"query": query, "k": k}
    response = requests.post(f"{API_BASE_URL}/retrieve", data=data)
    return response.json()

def call_openai_retrieve_api(query: str, model: str = "all-MiniLM-L6-v2", n_results: int = 5):
    data = {"query": query, "model": model, "n_results": n_results}
    response = requests.post(f"{API_BASE_URL}/v1/retrieve", data=data)
    return response.json()

def call_openai_embeddings_api(text: str, model: str = "text-embedding-ada-002", 
                              openai_api_key: str = None, openai_base_url: str = None):
    data = {
        "model": model,
        "input": text,
        "openai_api_key": openai_api_key,
        "openai_base_url": openai_base_url
    }
    response = requests.post(f"{API_BASE_URL}/v1/embeddings", data=data)
    return response.json()

def get_system_info_api():
    response = requests.get(f"{API_BASE_URL}/system_info")
    return response.json()

def get_file_info_api():
    response = requests.get(f"{API_BASE_URL}/file_info")
    return response.json()

def get_capabilities_api():
    response = requests.get(f"{API_BASE_URL}/capabilities")
    return response.json()

def download_file(url: str, filename: str):
    response = requests.get(f"{API_BASE_URL}{url}")
    return response.content

# ---------- Minimal DB Test Client (single table) ----------
def db_test_connection_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/test_connection", data=payload).json()

def db_list_tables_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/list_tables", data=payload).json()

def db_import_one_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/import_one", data=payload).json()

# ---------- Large File Helper Functions ----------
def is_large_file(file_size: int, threshold_mb: int = 100) -> bool:
    """Check if file is considered large"""
    return file_size > threshold_mb * 1024 * 1024

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def handle_file_upload(uploaded_file):
    """
    Safely handle file uploads by streaming to disk (no memory loading)
    Returns temporary file path and file info
    """
    # Create temporary file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        # Stream the uploaded file directly to disk
        shutil.copyfileobj(uploaded_file, tmp_file)
        temp_path = tmp_file.name
    
    # Get file size from disk
    file_size = os.path.getsize(temp_path)
    file_size_str = format_file_size(file_size)
    
    file_info = {
        "name": uploaded_file.name,
        "size": file_size_str,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Temporary storage",
        "temp_path": temp_path
    }
    
    return temp_path, file_info

# ---------- Streamlit App ----------
st.set_page_config(page_title="Chunking Optimizer", layout="wide", page_icon="📦")

# Custom header
st.markdown("""
<div style="background: linear-gradient(45deg, #FF8C00, #FFA500); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5em;">📦 Chunking Optimizer v2.0</h1>
    <p style="color: white; text-align: center; margin: 10px 0 0 0; font-size: 1.2em;">Advanced Text Processing + 1GB File Support + OpenAI API</p>
</div>
""", unsafe_allow_html=True)

# Session state
if "api_results" not in st.session_state:
    st.session_state.api_results = None
if "current_mode" not in st.session_state:
    st.session_state.current_mode = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "retrieval_results" not in st.session_state:
    st.session_state.retrieval_results = None
if "process_status" not in st.session_state:
    st.session_state.process_status = {
        "preprocessing": "pending",
        "chunking": "pending", 
        "embedding": "pending",
        "storage": "pending",
        "retrieval": "pending"
    }
if "process_timings" not in st.session_state:
    st.session_state.process_timings = {}
if "file_info" not in st.session_state:
    st.session_state.file_info = {}
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "column_types" not in st.session_state:
    st.session_state.column_types = {}
if "preview_df" not in st.session_state:
    st.session_state.preview_df = None
if "text_processing_option" not in st.session_state:
    st.session_state.text_processing_option = "none"
if "preview_updated" not in st.session_state:
    st.session_state.preview_updated = False
if "use_openai" not in st.session_state:
    st.session_state.use_openai = False
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "openai_base_url" not in st.session_state:
    st.session_state.openai_base_url = ""
if "process_large_files" not in st.session_state:
    st.session_state.process_large_files = True
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None

# Sidebar with process tracking and system info
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(45deg, #FF8C00, #FFA500); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: white; text-align: center; margin: 0;">Process Tracker</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # API connection test
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.success("✅ API Connected")
        
        # Show capabilities
        capabilities = get_capabilities_api()
        if capabilities.get('large_file_support'):
            st.info("🚀 1GB+ File Support Enabled")
        if capabilities.get('openai_compatible_endpoints'):
            st.info("🤖 OpenAI API Compatible")
            
    except:
        st.error("❌ API Not Connected")
    
    st.markdown("---")
    
    # OpenAI Configuration
    with st.expander("🤖 OpenAI Configuration"):
        st.session_state.use_openai = st.checkbox("Use OpenAI API", value=st.session_state.use_openai)
        
        if st.session_state.use_openai:
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", 
                                                          value=st.session_state.openai_api_key,
                                                          type="password",
                                                          help="Your OpenAI API key")
            st.session_state.openai_base_url = st.text_input("OpenAI Base URL (optional)", 
                                                           value=st.session_state.openai_base_url,
                                                           placeholder="https://api.openai.com/v1",
                                                           help="Custom OpenAI-compatible API endpoint")
            
            if st.session_state.openai_api_key:
                st.success("✅ OpenAI API Configured")
            else:
                st.warning("⚠️ Please enter OpenAI API Key")
    
    # Large File Configuration
    with st.expander("💾 Large File Settings"):
        st.session_state.process_large_files = st.checkbox(
            "Enable Large File Processing", 
            value=st.session_state.process_large_files,
            help="Process files larger than 100MB in batches to avoid memory issues"
        )
        
        if st.session_state.process_large_files:
            st.info("""
            **Large File Features:**
            - Direct disk streaming (no memory overload)
            - Batch processing for memory efficiency
            - Automatic chunking for files >100MB
            - Progress tracking for large datasets
            - Support for 1GB+ files
            """)
    
    # Minimal DB single-table test
    with st.expander("🗄️ Database (single-table test)"):
        db_type = st.selectbox("Type", ["mysql", "postgresql"], key="db_type_test")
        host = st.text_input("Host", "localhost", key="db_host_test")
        port = st.number_input("Port", 1, 65535, 3306 if db_type == "mysql" else 5432, key="db_port_test")
        username = st.text_input("Username", key="db_user_test")
        password = st.text_input("Password", type="password", key="db_pass_test")
        database = st.text_input("Database", key="db_name_test")
        
        processing_mode = st.selectbox(
            "Processing Mode:",
            ["fast", "config1", "deep"],
            key="db_processing_mode"
        )
        
        if st.button("🔌 Test", key="db_btn_test_conn"):
            res = db_test_connection_api({
                "db_type": db_type,
                "host": host,
                "port": port,
                "username": username,
                "password": password,
                "database": database,
            })
            st.write(res)
        if st.button("📋 Tables", key="db_btn_list_tables"):
            res = db_list_tables_api({
                "db_type": db_type,
                "host": host,
                "port": port,
                "username": username,
                "password": password,
                "database": database,
            })
            st.session_state["db_tables_test"] = res.get("tables", [])
            st.write(res)
        tables = st.session_state.get("db_tables_test", [])
        if tables:
            table_name = st.selectbox("Table", tables, key="db_table_select_test")
            if st.button("📥 Import One", key="db_btn_import_one"):
                res = db_import_one_api({
                    "db_type": db_type,
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "database": database,
                    "table_name": table_name,
                    "processing_mode": processing_mode,
                    "use_openai": st.session_state.use_openai,
                    "openai_api_key": st.session_state.openai_api_key,
                    "openai_base_url": st.session_state.openai_base_url
                })
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.success(f"Imported and processed via {processing_mode.upper()} pipeline")
                    st.session_state.api_results = res
                    st.session_state.current_mode = processing_mode
    
    # Process steps display
    st.markdown("### ⚙️ Processing Steps")
    
    steps = [
        ("preprocessing", "🧹 Preprocessing"),
        ("chunking", "📦 Chunking"), 
        ("embedding", "🤖 Embedding"),
        ("storage", "💾 Vector DB"),
        ("retrieval", "🔍 Retrieval")
    ]
    
    for step_key, step_name in steps:
        status = st.session_state.process_status.get(step_key, "pending")
        timing = st.session_state.process_timings.get(step_key, "")
        
        if status == "completed":
            icon = "✅"
            color = "completed"
            timing_display = f"({timing})" if timing else ""
        elif status == "running":
            icon = "🟠"
            color = "running"
            timing_display = ""
        else:
            icon = "⚪"
            color = "pending"
            timing_display = ""
        
        st.markdown(f"""
        <div class="process-step {color}">
            {icon} <strong>{step_name}</strong> {timing_display}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Information
    st.markdown("### 💻 System Information")
    try:
        system_info = get_system_info_api()
        st.write(f"**Memory Usage:** {system_info.get('memory_usage', 'N/A')}")
        st.write(f"**Available Memory:** {system_info.get('available_memory', 'N/A')}")
        st.write(f"**Total Memory:** {system_info.get('total_memory', 'N/A')}")
        if system_info.get('large_file_support'):
            st.write(f"**Max File Size:** {system_info.get('max_recommended_file_size', 'N/A')}")
    except:
        st.write("**Memory Usage:** N/A")
        st.write("**Available Memory:** N/A")
        st.write("**Total Memory:** N/A")
    
    # File Information
    st.markdown("### 📁 File Information")
    if st.session_state.file_info:
        file_info = st.session_state.file_info
        st.write(f"**File Name:** {file_info.get('name', 'N/A')}")
        st.write(f"**File Size:** {file_info.get('size', 'N/A')}")
        st.write(f"**Upload Time:** {file_info.get('upload_time', 'N/A')}")
        if file_info.get('large_file_processed'):
            st.success("✅ Large File Optimized")
        if file_info.get('large_file_sampled'):
            st.info("🔍 Large File Sampled")
    else:
        try:
            file_info = get_file_info_api()
            if file_info and 'filename' in file_info:
                st.write(f"**File Name:** {file_info.get('filename', 'N/A')}")
                st.write(f"**File Size:** {file_info.get('file_size', 0) / 1024:.2f} KB")
                st.write(f"**Upload Time:** {file_info.get('upload_time', 'N/A')}")
                st.write(f"**File Location:** Backend storage")
        except:
            st.write("**File Info:** Not available")
    
    st.markdown("---")
    
    if st.session_state.api_results:
        st.markdown("### 📊 Last Results")
        result = st.session_state.api_results
        st.write(f"**Mode:** {result.get('mode', 'N/A')}")
        if 'summary' in result:
            st.write(f"**Chunks:** {result['summary'].get('chunks', 'N/A')}")
            st.write(f"**Storage:** {result['summary'].get('stored', 'N/A')}")
            st.write(f"**Model:** {result['summary'].get('embedding_model', 'N/A')}")
            if 'conversion_results' in result['summary']:
                conv_results = result['summary']['conversion_results']
                if conv_results:
                    st.write(f"**Type Conversions:** {len(conv_results.get('successful', []))} successful")
            if result['summary'].get('retrieval_ready'):
                st.success("🔍 Retrieval Ready")
            if result['summary'].get('large_file_processed'):
                st.success("🚀 Large File Optimized")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        # Clean up temporary files
        if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
            os.unlink(st.session_state.temp_file_path)
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Mode selection
st.markdown("## 🎯 Choose Processing Mode")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⚡ Fast Mode", use_container_width=True):
        st.session_state.current_mode = "fast"
        st.session_state.process_status = {k: "pending" for k in st.session_state.process_status}
with col2:
    if st.button("⚙️ Config-1 Mode", use_container_width=True):
        st.session_state.current_mode = "config1"
        st.session_state.process_status = {k: "pending" for k in st.session_state.process_status}
with col3:
    if st.button("🔬 Deep Config Mode", use_container_width=True):
        st.session_state.current_mode = "deep"
        st.session_state.process_status = {k: "pending" for k in st.session_state.process_status}

if st.session_state.current_mode:
    st.success(f"Selected: **{st.session_state.current_mode.upper()} MODE**")

# Mode-specific processing
if st.session_state.current_mode:
    if st.session_state.current_mode == "fast":
        st.markdown("### ⚡ Fast Mode Configuration")
        
        # Input source selection
        input_source = st.radio("Select Input Source:", ["📁 Upload CSV File", "🗄️ Database Import"], key="fast_input_source")
        
        if input_source == "📁 Upload CSV File":
            st.markdown("#### 📤 Upload CSV File")
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="fast_file_upload")
            
            if uploaded_file is not None:
                # Use filesystem upload method
                with st.spinner("🔄 Streaming file to disk..."):
                    temp_path, file_info = handle_file_upload(uploaded_file)
                    st.session_state.temp_file_path = temp_path
                    st.session_state.file_info = file_info
                
                file_size_str = file_info["size"]
                
                # Check if file is large
                if is_large_file(os.path.getsize(temp_path)):
                    st.markdown(f"""
                    <div class="large-file-warning">
                        <strong>🚀 Large File Detected: {file_size_str}</strong><br>
                        Large file processing is {'ENABLED' if st.session_state.process_large_files else 'DISABLED'}<br>
                        <em>File streamed to disk - no memory overload</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(f"✅ **{uploaded_file.name}** loaded! ({file_size_str})")
                
            use_db_config = None
            
        else:  # Database Import
            st.markdown("#### 🗄️ Database Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                db_type = st.selectbox("Database Type", ["mysql", "postgresql"], key="fast_db_type")
                host = st.text_input("Host", "localhost", key="fast_host")
                port = st.number_input("Port", 1, 65535, 3306 if db_type == "mysql" else 5432, key="fast_port")
            
            with col2:
                username = st.text_input("Username", key="fast_username")
                password = st.text_input("Password", type="password", key="fast_password")
                database = st.text_input("Database", key="fast_database")
            
            # Test connection and get tables
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Test Connection", key="fast_test_conn"):
                    res = db_test_connection_api({
                        "db_type": db_type,
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "database": database,
                    })
                    if res.get("status") == "success":
                        st.success("✅ Connection successful!")
                    else:
                        st.error(f"❌ Connection failed: {res.get('message', 'Unknown error')}")
            
            with col2:
                if st.button("📋 List Tables", key="fast_list_tables"):
                    res = db_list_tables_api({
                        "db_type": db_type,
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "database": database,
                    })
                    st.session_state["fast_db_tables"] = res.get("tables", [])
                    if st.session_state["fast_db_tables"]:
                        st.success(f"✅ Found {len(st.session_state['fast_db_tables'])} tables")
                    else:
                        st.warning("⚠️ No tables found")
            
            tables = st.session_state.get("fast_db_tables", [])
            if tables:
                table_name = st.selectbox("Select Table", tables, key="fast_table_select")
                use_db_config = {
                    "use_db": True,
                    "db_type": db_type,
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "database": database,
                    "table_name": table_name
                }
            else:
                use_db_config = None
                st.info("👆 Test connection and list tables first")
        
        st.markdown("""<div class="custom-card">
            <div class="card-title">Auto-optimized Pipeline</div>
            <div class="card-content">
                • Auto preprocessing (drop nulls + lowercase + remove delimiters + remove whitespace)<br>
                • Semantic clustering chunking<br>
                • MiniLM or OpenAI embeddings<br>
                • FAISS storage<br>
                • 1GB+ file support with disk streaming<br>
                • SQL Database support with chunked imports
            </div>
        </div>""", unsafe_allow_html=True)
        
        run_enabled = (
            (input_source == "📁 Upload CSV File" and st.session_state.get('temp_file_path') is not None) or
            (input_source == "🗄️ Database Import" and use_db_config is not None)
        )
        
        if st.button("🚀 Run Fast Pipeline", type="primary", use_container_width=True, disabled=not run_enabled):
            with st.spinner("Running Fast Mode pipeline..."):
                try:
                    if input_source == "📁 Upload CSV File":
                        result = call_fast_api(
                            st.session_state.temp_file_path,
                            st.session_state.file_info["name"],
                            "sqlite",
                            use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files
                        )
                    else:
                        result = call_fast_api(
                            None, None, "sqlite", use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files
                        )
                    
                    # Update process status
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "completed"
                        st.session_state.process_timings[step] = "Completed"
                    
                    st.session_state.api_results = result
                    
                    # Show special notifications
                    if 'summary' in result and result['summary'].get('large_file_processed'):
                        st.success("✅ Large file processed efficiently with disk streaming!")
                    else:
                        st.success("✅ Fast pipeline completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")
                finally:
                    # Clean up temporary file
                    if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
                        os.unlink(st.session_state.temp_file_path)
                        st.session_state.temp_file_path = None

    elif st.session_state.current_mode == "config1":
        st.markdown("### ⚙️ Config-1 Mode Configuration")
        
        # Input source selection
        input_source = st.radio("Select Input Source:", ["📁 Upload CSV File", "🗄️ Database Import"], key="config1_input_source")
        
        if input_source == "📁 Upload CSV File":
            st.markdown("#### 📤 Upload CSV File")
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="config1_file_upload")
            
            if uploaded_file is not None:
                # Use filesystem upload method
                with st.spinner("🔄 Streaming file to disk..."):
                    temp_path, file_info = handle_file_upload(uploaded_file)
                    st.session_state.temp_file_path = temp_path
                    st.session_state.file_info = file_info
                
                file_size_str = file_info["size"]
                
                # Check if file is large
                if is_large_file(os.path.getsize(temp_path)):
                    st.markdown(f"""
                    <div class="large-file-warning">
                        <strong>🚀 Large File Detected: {file_size_str}</strong><br>
                        Large file processing is {'ENABLED' if st.session_state.process_large_files else 'DISABLED'}<br>
                        <em>File streamed to disk - no memory overload</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(f"✅ **{uploaded_file.name}** loaded! ({file_size_str})")
                
            use_db_config = None
            
        else:  # Database Import
            st.markdown("#### 🗄️ Database Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                db_type = st.selectbox("Database Type", ["mysql", "postgresql"], key="config1_db_type")
                host = st.text_input("Host", "localhost", key="config1_host")
                port = st.number_input("Port", 1, 65535, 3306 if db_type == "mysql" else 5432, key="config1_port")
            
            with col2:
                username = st.text_input("Username", key="config1_username")
                password = st.text_input("Password", type="password", key="config1_password")
                database = st.text_input("Database", key="config1_database")
            
            # Test connection and get tables
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Test Connection", key="config1_test_conn"):
                    res = db_test_connection_api({
                        "db_type": db_type,
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "database": database,
                    })
                    if res.get("status") == "success":
                        st.success("✅ Connection successful!")
                    else:
                        st.error(f"❌ Connection failed: {res.get('message', 'Unknown error')}")
            
            with col2:
                if st.button("📋 List Tables", key="config1_list_tables"):
                    res = db_list_tables_api({
                        "db_type": db_type,
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "database": database,
                    })
                    st.session_state["config1_db_tables"] = res.get("tables", [])
                    if st.session_state["config1_db_tables"]:
                        st.success(f"✅ Found {len(st.session_state['config1_db_tables'])} tables")
                    else:
                        st.warning("⚠️ No tables found")
            
            tables = st.session_state.get("config1_db_tables", [])
            if tables:
                table_name = st.selectbox("Select Table", tables, key="config1_table_select")
                use_db_config = {
                    "use_db": True,
                    "db_type": db_type,
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "database": database,
                    "table_name": table_name
                }
            else:
                use_db_config = None
                st.info("👆 Test connection and list tables first")
        
        # Config-1 parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧹 Preprocessing")
            null_handling = st.selectbox("Null value handling", ["keep", "drop", "fill"], key="config1_null")
            fill_value = st.text_input("Fill value", "Unknown", key="config1_fill") if null_handling == "fill" else None
            
            st.markdown("#### 📦 Chunking")
            chunk_method = st.selectbox("Chunking method", ["fixed", "recursive", "semantic", "document"], key="config1_chunk")
            
            if chunk_method in ["fixed", "recursive"]:
                chunk_size = st.number_input("Chunk size", 100, 2000, 400, key="config1_size")
                overlap = st.number_input("Overlap", 0, 500, 50, key="config1_overlap")
        
        with col2:
            st.markdown("#### 🤖 Embedding")
            model_choice = st.selectbox("Embedding model", 
                                      ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "text-embedding-ada-002"],
                                      key="config1_model")
            
            st.markdown("#### 💾 Storage")
            storage_choice = st.selectbox("Vector storage", ["faiss", "chromadb"], key="config1_storage")
        
        st.info("ℹ️ Text preprocessing: Auto lowercase + remove delimiters + remove whitespace")
        
        run_enabled = (
            (input_source == "📁 Upload CSV File" and st.session_state.get('temp_file_path') is not None) or
            (input_source == "🗄️ Database Import" and use_db_config is not None)
        )
        
        if st.button("🚀 Run Config-1 Pipeline", type="primary", use_container_width=True, disabled=not run_enabled):
            with st.spinner("Running Config-1 pipeline..."):
                try:
                    config = {
                        "null_handling": null_handling,
                        "fill_value": fill_value if fill_value else "Unknown",
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 400,
                        "overlap": overlap if 'overlap' in locals() else 50,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                    }
                    
                    if input_source == "📁 Upload CSV File":
                        result = call_config1_api(
                            st.session_state.temp_file_path,
                            st.session_state.file_info["name"],
                            config,
                            use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files
                        )
                    else:
                        result = call_config1_api(
                            None, None, config, use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files
                        )
                    
                    # Mark all as completed
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "completed"
                        st.session_state.process_timings[step] = "Completed"
                    
                    st.session_state.api_results = result
                    
                    # Show special notifications
                    if 'summary' in result and result['summary'].get('large_file_processed'):
                        st.success("✅ Large file processed efficiently with disk streaming!")
                    else:
                        st.success("✅ Config-1 pipeline completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")
                finally:
                    # Clean up temporary file
                    if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
                        os.unlink(st.session_state.temp_file_path)
                        st.session_state.temp_file_path = None

    elif st.session_state.current_mode == "deep":
        st.markdown("### 🔬 Deep Config Mode Configuration")
        
        # Input source selection
        input_source = st.radio("Select Input Source:", ["📁 Upload CSV File", "🗄️ Database Import"], key="deep_input_source")
        
        if input_source == "📁 Upload CSV File":
            st.markdown("#### 📤 Upload CSV File")
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="deep_file_upload")
            
            if uploaded_file is not None:
                # Use filesystem upload method
                with st.spinner("🔄 Streaming file to disk..."):
                    temp_path, file_info = handle_file_upload(uploaded_file)
                    st.session_state.temp_file_path = temp_path
                    st.session_state.file_info = file_info
                
                file_size_str = file_info["size"]
                
                # Check if file is large
                if is_large_file(os.path.getsize(temp_path)):
                    st.markdown(f"""
                    <div class="large-file-warning">
                        <strong>🚀 Large File Detected: {file_size_str}</strong><br>
                        Large file processing is {'ENABLED' if st.session_state.process_large_files else 'DISABLED'}<br>
                        <em>File streamed to disk - no memory overload</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Read the CSV file for preview and column type analysis
                try:
                    df = pd.read_csv(temp_path)
                    st.session_state.current_df = df
                    # Initialize preview only once
                    if "preview_df" not in st.session_state or st.session_state.preview_df is None:
                        st.session_state.preview_df = df.head(5).copy()
                    st.success(f"✅ **{uploaded_file.name}** loaded! ({len(df)} rows, {len(df.columns)} columns, {file_size_str})")
                    
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                
            use_db_config = None
            
        else:  # Database Import
            st.markdown("#### 🗄️ Database Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                db_type = st.selectbox("Database Type", ["mysql", "postgresql"], key="deep_db_type")
                host = st.text_input("Host", "localhost", key="deep_host")
                port = st.number_input("Port", 1, 65535, 3306 if db_type == "mysql" else 5432, key="deep_port")
            
            with col2:
                username = st.text_input("Username", key="deep_username")
                password = st.text_input("Password", type="password", key="deep_password")
                database = st.text_input("Database", key="deep_database")
            
            # Test connection and get tables
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Test Connection", key="deep_test_conn"):
                    res = db_test_connection_api({
                        "db_type": db_type,
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "database": database,
                    })
                    if res.get("status") == "success":
                        st.success("✅ Connection successful!")
                    else:
                        st.error(f"❌ Connection failed: {res.get('message', 'Unknown error')}")
            
            with col2:
                if st.button("📋 List Tables", key="deep_list_tables"):
                    res = db_list_tables_api({
                        "db_type": db_type,
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "database": database,
                    })
                    st.session_state["deep_db_tables"] = res.get("tables", [])
                    if st.session_state["deep_db_tables"]:
                        st.success(f"✅ Found {len(st.session_state['deep_db_tables'])} tables")
                    else:
                        st.warning("⚠️ No tables found")
            
            tables = st.session_state.get("deep_db_tables", [])
            if tables:
                table_name = st.selectbox("Select Table", tables, key="deep_table_select")
                use_db_config = {
                    "use_db": True,
                    "db_type": db_type,
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "database": database,
                    "table_name": table_name
                }
                
            else:
                use_db_config = None
                st.info("👆 Test connection and list tables first")
        
        # Deep config parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧹 Enhanced Preprocessing Pipeline")
            
            # Step-by-step preprocessing workflow
            if input_source == "📁 Upload CSV File" and st.session_state.current_df is not None:
                df = st.session_state.current_df
                
                # Initialize preprocessing step
                if "preprocessing_step" not in st.session_state:
                    st.session_state.preprocessing_step = 0
                
                # Step 0: Default preprocessing (header validation, null representation cleanup)
                if st.session_state.preprocessing_step == 0:
                    st.markdown("**Step 1: Default Preprocessing**")
                    st.info("✅ Header validation and null representation cleanup")
                    
                    if st.button("Run Default Preprocessing", key="default_preprocessing"):
                        # Apply header validation and null cleanup
                        df.columns = validate_and_normalize_headers(df.columns)
                        df.replace(["", " ", "NA", "N/A", "NULL", "-", "--", "NaN"], pd.NA, inplace=True)
                        
                        # Normalize text columns
                        text_cols = df.select_dtypes(include=['object']).columns.tolist()
                        for col in text_cols:
                            df[col] = normalize_text_column(df[col])
                        
                        st.session_state.current_df = df
                        st.session_state.preprocessing_step = 1
                        st.rerun()
                
                # Step 1: Type conversion
                if st.session_state.preprocessing_step == 1:
                    st.markdown("**Step 2: Data Type Conversion**")
                    st.info("Convert column types before processing:")
                    
                    preview_df = df.head(5).copy()
                    column_types = st.session_state.get("column_types", {})
                    
                    # Display column headers with data type selection
                    for col in preview_df.columns:
                        current_type = str(preview_df[col].dtype)
                        default_idx = 0
                        type_options = ["keep", "text", "numeric", "datetime"]
                        
                        # Set default based on current conversion
                        if col in column_types:
                            try:
                                default_idx = type_options.index(column_types[col])
                            except ValueError:
                                default_idx = 0
                        
                        new_type = st.selectbox(
                            f"**{col}** › Current: `{current_type}`",
                            type_options,
                            index=default_idx,
                            key=f"col_type_{col}"
                        )
                        
                        if new_type != "keep":
                            column_types[col] = new_type
                            # Apply conversion to preview
                            try:
                                if new_type == 'text':
                                    preview_df[col] = preview_df[col].astype(str)
                                elif new_type == 'numeric':
                                    preview_df[col] = pd.to_numeric(preview_df[col], errors='coerce')
                                elif new_type == 'datetime':
                                    preview_df[col] = pd.to_datetime(preview_df[col], errors='coerce')
                            except Exception as e:
                                st.error(f"Error converting {col}: {str(e)}")
                        elif col in column_types:
                            # Remove from conversions if set to "keep"
                            del column_types[col]
                    
                    # Display the updated preview
                    st.dataframe(preview_df, use_container_width=True)
                    
                    st.session_state.column_types = column_types
                    
                    if column_types:
                        st.success(f"🎯 {len(column_types)} columns will be converted")
                    else:
                        st.info("No column type conversions selected")
                    
                    if st.button("Apply Type Conversion", key="apply_type_conversion"):
                        # Apply type conversions to full dataframe
                        for col, new_type in column_types.items():
                            try:
                                if new_type == 'text':
                                    df[col] = df[col].astype(str)
                                elif new_type == 'numeric':
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                elif new_type == 'datetime':
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
                            except Exception as e:
                                st.error(f"Error converting {col}: {str(e)}")
                        
                        st.session_state.current_df = df
                        st.session_state.preprocessing_step = 2
                        st.rerun()
                
                # Step 2: Null handling
                if st.session_state.preprocessing_step == 2:
                    st.markdown("**Step 3: Null Values Handling**")
                    
                    # Show null summary
                    num_nulls = df.isnull().sum()
                    null_percent = (num_nulls / len(df) * 100) if len(df) > 0 else 0
                    
                    null_summary = pd.DataFrame({
                        "column_name": df.columns,
                        "num_nulls": num_nulls.values,
                        "null_percent": null_percent.values if hasattr(null_percent, "values") else null_percent
                    })
                    
                    st.dataframe(null_summary)
                    
                    cols_with_nulls = df.columns[df.isnull().any()].tolist()
                    
                    if cols_with_nulls:
                        selected_col = st.selectbox("Select column to handle nulls", cols_with_nulls, key="null_column_select")
                        col_series = df[selected_col]
                        
                        # Define strategies based on dtype
                        if pd.api.types.is_numeric_dtype(col_series):
                            strategies = [
                                "Skip / Leave as is",
                                "Drop Rows",
                                "Fill with Mean",
                                "Fill with Median",
                                "Fill with Mode",
                                "Fill with Custom Value"
                            ]
                        elif pd.api.types.is_datetime64_any_dtype(col_series):
                            strategies = [
                                "Skip / Leave as is",
                                "Drop Rows",
                                "Fill with Mode",
                                "Fill with Custom Value"
                            ]
                        else:
                            strategies = [
                                "Skip / Leave as is",
                                "Drop Rows",
                                "Fill with Mode",
                                "Fill with Custom Value"
                            ]
                        
                        strategy = st.radio("Select handling strategy:", strategies, key=f"strategy_{selected_col}")
                        
                        custom_val = None
                        if strategy == "Fill with Custom Value":
                            custom_val = st.text_input(
                                f"Enter custom value for '{selected_col}':",
                                key=f"custom_{selected_col}"
                            )
                        
                        if st.button("Apply Null Handling", key=f"apply_null_{selected_col}"):
                            df_copy = df.copy()
                            
                            if strategy == "Skip / Leave as is":
                                st.info(f"Skipped null handling for '{selected_col}'. No changes made.")
                            else:
                                try:
                                    if strategy == "Drop Rows":
                                        df_copy = df_copy.dropna(subset=[selected_col])
                                    elif strategy == "Fill with Mean" and pd.api.types.is_numeric_dtype(df_copy[selected_col]):
                                        df_copy[selected_col] = df_copy[selected_col].fillna(df_copy[selected_col].mean())
                                    elif strategy == "Fill with Median" and pd.api.types.is_numeric_dtype(df_copy[selected_col]):
                                        df_copy[selected_col] = df_copy[selected_col].fillna(df_copy[selected_col].median())
                                    elif strategy == "Fill with Mode":
                                        mode_val = df_copy[selected_col].mode()
                                        if not mode_val.empty:
                                            df_copy[selected_col] = df_copy[selected_col].fillna(mode_val.iloc[0])
                                    elif strategy == "Fill with Custom Value" and custom_val:
                                        if pd.api.types.is_datetime64_any_dtype(df_copy[selected_col]):
                                            parsed_val = pd.to_datetime(custom_val)
                                            df_copy[selected_col] = df_copy[selected_col].fillna(parsed_val)
                                        elif pd.api.types.is_numeric_dtype(df_copy[selected_col]):
                                            parsed_val = float(custom_val)
                                            df_copy[selected_col] = df_copy[selected_col].fillna(parsed_val)
                                        else:
                                            df_copy[selected_col] = df_copy[selected_col].fillna(custom_val)
                                except Exception as e:
                                    st.error(f"Error during null handling: {e}")
                                    st.stop()
                                
                                st.success(f"Null handling applied on '{selected_col}' using '{strategy}'.")
                            
                            st.session_state.current_df = df_copy
                            st.rerun()
                        
                        # Check if all nulls are handled
                        if df.isnull().sum().sum() == 0 or st.button("Proceed to Duplicates", key="proceed_to_duplicates"):
                            st.session_state.preprocessing_step = 3
                            st.rerun()
                    else:
                        st.info("No null values detected; moving to next step.")
                        if st.button("Proceed to Duplicates", key="proceed_to_duplicates_no_nulls"):
                            st.session_state.preprocessing_step = 3
                            st.rerun()
                
                # Step 3: Duplicate handling
                if st.session_state.preprocessing_step == 3:
                    st.markdown("**Step 4: Duplicate Handling**")
                    
                    dup_count = df.duplicated().sum()
                    st.write(f"**Duplicate Rows: {dup_count}**")
                    
                    if dup_count > 0:
                        if st.button("Remove Duplicate Rows", key="remove_duplicates"):
                            df = df.drop_duplicates(keep="first")
                            st.session_state.current_df = df
                            st.session_state.preprocessing_step = 4
                            st.rerun()
                        
                        if st.button("Keep Duplicate Rows", key="keep_duplicates"):
                            st.session_state.preprocessing_step = 4
                            st.rerun()
                    else:
                        st.info("No duplicate rows detected; skipping duplicate handling.")
                        if st.button("Proceed to Text Processing", key="proceed_to_text"):
                            st.session_state.preprocessing_step = 4
                            st.rerun()
                
                # Step 4: Text processing
                if st.session_state.preprocessing_step == 4:
                    st.markdown("**Step 5: Text Processing**")
                    
                    remove_stopwords = st.checkbox("Remove stopwords", key="deep_stop")
                    
                    # Radio button for stemming vs lemmatization
                    text_processing_option = st.radio(
                        "Advanced text processing:",
                        ["none", "stem", "lemmatize"],
                        index=0,
                        key="deep_text_processing"
                    )
                    
                    if st.button("Apply Text Processing", key="apply_text_processing"):
                        if remove_stopwords:
                            df, _ = remove_stopwords_from_text_columns(df, remove_stopwords=True)
                        
                        if text_processing_option in ["stem", "lemmatize"]:
                            df, _ = apply_text_normalization(df, text_processing_option)
                        
                        st.session_state.current_df = df
                        st.session_state.preprocessing_step = 5
                        st.rerun()
                
                # Step 5: Final preview
                if st.session_state.preprocessing_step == 5:
                    st.markdown("**✅ Preprocessing Complete!**")
                    st.markdown("**Final Data Preview:**")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Show file metadata
                    file_meta = {
                        'file_source': uploaded_file.name if uploaded_file else 'dataframe_input',
                        'num_rows': df.shape[0],
                        'num_columns': df.shape[1],
                        'shape': df.shape,
                        'upload_time': pd.Timestamp.now().isoformat()
                    }
                    
                    st.markdown("**File Metadata:**")
                    st.json(file_meta)
                    
                    # Show numeric column metadata
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        st.markdown("**Numeric Column Metadata:**")
                        numeric_metadata = []
                        for col in numeric_cols:
                            numeric_metadata.append({
                                'column_name': col,
                                'count': int(df[col].count()),
                                'mean': float(df[col].mean()) if not df[col].empty else 0.0,
                                'std': float(df[col].std()) if not df[col].empty else 0.0,
                                'min': float(df[col].min()) if not df[col].empty else 0.0,
                                'max': float(df[col].max()) if not df[col].empty else 0.0,
                            })
                        st.json(numeric_metadata)
                    
                    if st.button("Reset Preprocessing", key="reset_preprocessing"):
                        st.session_state.preprocessing_step = 0
                        st.rerun()
            
            # Legacy preprocessing options for database imports
            else:
                st.markdown("#### 🧹 Preprocessing")
                null_handling = st.selectbox("Null value handling", ["keep", "drop", "fill"], key="deep_null")
                fill_value = st.text_input("Fill value", "Unknown", key="deep_fill") if null_handling == "fill" else None
                
                st.markdown("#### 🧠 Text Processing")
                remove_stopwords = st.checkbox("Remove stopwords", key="deep_stop")
                lowercase = st.checkbox("Convert to lowercase + clean text", value=True, key="deep_lower")
                
                # Radio button for stemming vs lemmatization (mutually exclusive)
                text_processing_option = st.radio(
                    "Advanced text processing:",
                    ["none", "stemming", "lemmatization"],
                    index=0,
                    key="deep_text_processing"
                )
            
        with col2:
            st.markdown("#### 📦 Chunking")
            chunk_method = st.selectbox("Chunking method", ["fixed", "recursive", "semantic", "document"], key="deep_chunk")
            
            if chunk_method in ["fixed", "recursive"]:
                chunk_size = st.number_input("Chunk size", 100, 2000, 400, key="deep_size")
                overlap = st.number_input("Overlap", 0, 500, 50, key="deep_overlap")
            
            # Document chunking column selection - show for both file and database
            if chunk_method == "document":
                if st.session_state.current_df is not None:
                    available_columns = st.session_state.current_df.columns.tolist()
                    document_key_column = st.selectbox(
                        "Select column for grouping:",
                        available_columns,
                        key="deep_document_column"
                    )
                    st.info(f"Chunks will be grouped by: **{document_key_column}**")
                else:
                    document_key_column = st.text_input(
                        "Enter column name for grouping:",
                        key="deep_document_column_text"
                    )
                    if document_key_column:
                        st.info(f"Chunks will be grouped by: **{document_key_column}**")
            
            st.markdown("#### 🤖 Embedding & Storage")
            model_choice = st.selectbox("Embedding model", 
                                      ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "text-embedding-ada-002"],
                                      key="deep_model")
            storage_choice = st.selectbox("Vector storage", ["faiss", "chromadb"], key="deep_storage")
        
        run_enabled = (
            (input_source == "📁 Upload CSV File" and st.session_state.get('temp_file_path') is not None) or
            (input_source == "🗄️ Database Import" and use_db_config is not None)
        )
        
        if st.button("🚀 Run Deep Config Pipeline", type="primary", use_container_width=True, disabled=not run_enabled):
            with st.spinner("Running Deep Config pipeline..."):
                try:
                    config = {
                        "null_handling": null_handling,
                        "fill_value": fill_value if fill_value else "Unknown",
                        "remove_stopwords": remove_stopwords,
                        "lowercase": lowercase,
                        "text_processing_option": text_processing_option,
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 400,
                        "overlap": overlap if 'overlap' in locals() else 50,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                    }
                    
                    # Add column types only for file uploads (not for database)
                    if input_source == "📁 Upload CSV File":
                        config["column_types"] = json.dumps(st.session_state.column_types)
                    
                    # Add document key column for document chunking
                    if chunk_method == "document":
                        if 'document_key_column' in locals() and document_key_column:
                            config["document_key_column"] = document_key_column
                        elif st.session_state.current_df is not None and len(st.session_state.current_df.columns) > 0:
                            # Use first column as default
                            config["document_key_column"] = st.session_state.current_df.columns[0]
                    
                    if input_source == "📁 Upload CSV File":
                        result = call_deep_api(
                            st.session_state.temp_file_path,
                            st.session_state.file_info["name"],
                            config,
                            use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files
                        )
                    else:
                        result = call_deep_api(
                            None, None, config, use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files
                        )
                    
                    # Mark all as completed
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "completed"
                        st.session_state.process_timings[step] = "Completed"
                    
                    st.session_state.api_results = result
                    
                    # Show conversion results if available
                    if 'summary' in result and 'conversion_results' in result['summary']:
                        conv_results = result['summary']['conversion_results']
                        if conv_results:
                            st.success(f"✅ Column type conversion: {len(conv_results.get('successful', []))} successful")
                            if conv_results.get('failed'):
                                st.warning(f"⚠️ {len(conv_results['failed'])} conversions failed")
                    
                    # Show special notifications
                    if 'summary' in result and result['summary'].get('large_file_processed'):
                        st.success("✅ Large file processed efficiently with disk streaming!")
                    else:
                        st.success("✅ Deep Config pipeline completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")
                finally:
                    # Clean up temporary file
                    if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
                        os.unlink(st.session_state.temp_file_path)
                        st.session_state.temp_file_path = None

# OpenAI API Testing Section
st.markdown("---")
st.markdown("## 🤖 OpenAI API Testing")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔤 Test Embeddings")
    embed_text = st.text_area("Text to embed:", "Hello, world! This is a test.", key="embed_text")
    embed_model = st.selectbox("Embedding Model", 
                              ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
                              key="embed_model")
    
    if st.button("Generate Embeddings", key="test_embed"):
        with st.spinner("Generating embeddings..."):
            try:
                result = call_openai_embeddings_api(
                    embed_text, 
                    embed_model,
                    st.session_state.openai_api_key if st.session_state.use_openai else None,
                    st.session_state.openai_base_url if st.session_state.use_openai else None
                )
                
                if "data" in result:
                    st.success(f"✅ Generated {len(result['data'])} embedding(s)")
                    st.json(result)
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Embedding error: {str(e)}")

with col2:
    st.markdown("### 🔍 Test Retrieval")
    retrieve_query = st.text_input("Retrieval query:", "test data", key="retrieve_query")
    retrieve_model = st.selectbox("Retrieval Model", 
                                 ["all-MiniLM-L6-v2", "text-embedding-ada-002"],
                                 key="retrieve_model")
    n_results = st.slider("Number of results", 1, 10, 3, key="n_results")
    
    if st.button("Test Retrieval", key="test_retrieve"):
        with st.spinner("Searching..."):
            try:
                result = call_openai_retrieve_api(retrieve_query, retrieve_model, n_results)
                
                if "data" in result:
                    st.success(f"✅ Found {len(result['data'])} results")
                    for res in result['data']:
                        st.write(f"**Rank {res['rank']}** (Score: {res['score']:.3f}):")
                        st.write(f"{res['content'][:200]}...")
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")

# Vector Retrieval Section
if st.session_state.api_results and st.session_state.api_results.get('summary', {}).get('retrieval_ready'):
    st.markdown("---")
    st.markdown("## 🔍 Semantic Search (Vector DB)")
    st.markdown("Search for similar content using semantic similarity")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        vector_query = st.text_input("Enter semantic search query:", placeholder="Search for similar content...", key="vector_query")
    with col2:
        k = st.slider("Top K results", 1, 10, 3, key="vector_k")
    
    if vector_query:
        with st.spinner("Searching..."):
            try:
                st.session_state.process_status["retrieval"] = "running"
                retrieval_result = call_retrieve_api(vector_query, k)
                st.session_state.process_status["retrieval"] = "completed"
                st.session_state.retrieval_results = retrieval_result
                
                if "error" in retrieval_result:
                    st.error(f"Retrieval error: {retrieval_result['error']}")
                else:
                    st.success(f"✅ Found {len(retrieval_result['results'])} results")
                    
                    for result in retrieval_result['results']:
                        similarity_color = "#28a745" if result['similarity'] > 0.7 else "#ffc107" if result['similarity'] > 0.4 else "#dc3545"
                        
                        st.markdown(f"""
                        <div style="background: #2d2d2d; border: 1px solid #444; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {similarity_color};">
                            <h4 style="margin: 0 0 10px 0; color: {similarity_color};">
                                Rank #{result['rank']} (Similarity: {result['similarity']:.3f})
                            </h4>
                            <p style="margin: 0; color: #cccccc; font-size: 0.9em;">{result['content'][:300]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")

# Export Section
if st.session_state.api_results:
    st.markdown("---")
    st.markdown("## 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📥 Download Chunks")
        if st.button("📄 Export Chunks as TXT", use_container_width=True):
            try:
                chunks_content = download_file("/export/chunks", "chunks.txt")
                st.download_button(
                    label="⬇️ Download Chunks",
                    data=chunks_content,
                    file_name="chunks.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error exporting chunks: {str(e)}")
    
    with col2:
        st.markdown("#### 📥 Download Embeddings")
        if st.button("🔢 Export Embeddings as NPY", use_container_width=True):
            try:
                embeddings_content = download_file("/export/embeddings", "embeddings.npy")
                st.download_button(
                    label="⬇️ Download Embeddings",
                    data=embeddings_content,
                    file_name="embeddings.npy",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error exporting embeddings: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>📦 Chunking Optimizer v2.0 • FastAPI + Streamlit • 1GB+ File Support • OpenAI API Compatible</p>
    <p><strong>🚀 Enhanced with Disk Streaming for Large Files</strong></p>
</div>
""", unsafe_allow_html=True)