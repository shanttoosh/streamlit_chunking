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