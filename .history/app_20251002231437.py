# app.py (Streamlit Frontend)
import streamlit as st
import pandas as pd
import requests
import io
import time
import base64
import os
from datetime import datetime

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
</style>
""", unsafe_allow_html=True)

# ---------- API Client Functions ----------
def call_fast_api(file_content: bytes, filename: str, db_type: str):
    files = {"file": (filename, file_content, "text/csv")}
    data = {"db_type": db_type}
    response = requests.post(f"{API_BASE_URL}/run_fast", files=files, data=data)
    return response.json()

def call_config1_api(file_content: bytes, filename: str, config: dict):
    files = {"file": (filename, file_content, "text/csv")}
    data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
    response = requests.post(f"{API_BASE_URL}/run_config1", files=files, data=data)
    return response.json()

def call_deep_api(file_content: bytes, filename: str, config: dict):
    files = {"file": (filename, file_content, "text/csv")}
    data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
    response = requests.post(f"{API_BASE_URL}/run_deep", files=files, data=data)
    return response.json()

def call_retrieve_api(query: str, k: int = 5):
    data = {"query": query, "k": k}
    response = requests.post(f"{API_BASE_URL}/retrieve", data=data)
    return response.json()

    

def get_system_info_api():
    response = requests.get(f"{API_BASE_URL}/system_info")
    return response.json()

def get_file_info_api():
    response = requests.get(f"{API_BASE_URL}/file_info")
    return response.json()

def download_file(url: str, filename: str):
    response = requests.get(f"{API_BASE_URL}{url}")
    return response.content

# ---------- Database Integration API Client ----------
def db_test_connection_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/test_connection", data=payload).json()

def db_list_tables_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/list_tables", data=payload).json()

def db_import_one_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/import_one", data=payload).json()

def db_run_config1_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/run_config1", data=payload).json()

def db_run_deep_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/run_deep", data=payload).json()

# ---------- Streamlit App ----------
st.set_page_config(page_title="Chunking Optimizer", layout="wide", page_icon="📦")

# Custom header
st.markdown("""
<div style="background: linear-gradient(45deg, #FF8C00, #FFA500); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5em;">📦 Chunking Optimizer</h1>
    <p style="color: white; text-align: center; margin: 10px 0 0 0; font-size: 1.2em;">Advanced Text Processing + SQL Query Support</p>
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
    except:
        st.error("❌ API Not Connected")
    
    st.markdown("---")
    
    # Database Integration with all modes
    with st.expander("🗄️ Database Integration"):
        db_type = st.selectbox("Type", ["mysql", "postgresql"], key="db_type_test")
        host = st.text_input("Host", "localhost", key="db_host_test")
        port = st.number_input("Port", 1, 65535, 3306 if db_type == "mysql" else 5432, key="db_port_test")
        username = st.text_input("Username", key="db_user_test")
        password = st.text_input("Password", type="password", key="db_pass_test")
        database = st.text_input("Database", key="db_name_test")
        
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
            
            # Processing mode selection
            db_mode = st.selectbox("Processing Mode", 
                                 ["Fast (Auto)", "Config-1 (Customizable)", "Deep (Advanced)"], 
                                 key="db_mode_select")
            
            # Base payload for all modes
            base_db_payload = {
                "db_type": db_type,
                "host": host,
                "port": port,
                "username": username,
                "password": password,
                "database": database,
                "table_name": table_name,
            }
            
            if db_mode == "Fast (Auto)":
                if st.button("📥 Run Fast Mode", key="db_btn_import_one"):
                    res = db_import_one_api(base_db_payload)
                    if "error" in res:
                    st.error(res["error"])
                else:
                    st.success("Imported and processed via Fast pipeline")
                    st.session_state.api_results = res
    
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
        st.write(f"**File Location:** {file_info.get('location', 'Temporary storage')}")
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
            if result['summary'].get('retrieval_ready'):
                st.success("🔍 Retrieval Ready")
            
    
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Mode selection
st.markdown("## 🎯 Choose Processing Mode")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⚡ Fast Mode", use_container_width=True):
        st.session_state.current_mode = "fast"
        # Reset process status
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

# File upload
st.markdown("### 📤 Upload CSV File")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    # Store file information
    st.session_state.file_info = {
        "name": uploaded_file.name,
        "size": f"{uploaded_file.size / 1024:.2f} KB",
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Temporary storage"
    }
    st.success(f"✅ **{uploaded_file.name}** loaded!")
    
    # REMOVED TABLE PREVIEW as requested

# Mode-specific processing
if st.session_state.current_mode and st.session_state.uploaded_file:
    if st.session_state.current_mode == "fast":
        st.markdown("### ⚡ Fast Mode Configuration")
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">Auto-optimized Pipeline</div>
            <div class="card-content">
                • Auto preprocessing (drop nulls)<br>
                • Semantic clustering chunking<br>
                • MiniLM embeddings<br>
                • FAISS storage<br>
                • SQL Database
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        
        
        if st.button("🚀 Run Fast Pipeline", type="primary", use_container_width=True):
            # Update process status
            st.session_state.process_status["preprocessing"] = "running"
            
            with st.spinner("Running Fast Mode pipeline..."):
                try:
                    # Update timings as we go
                    start_time = time.time()
                    result = call_fast_api(
                        st.session_state.uploaded_file.getvalue(),
                        st.session_state.uploaded_file.name,
                        "sqlite"
                    )
                    st.session_state.process_timings["preprocessing"] = f"{time.time() - start_time:.2f}s"
                    st.session_state.process_status["preprocessing"] = "completed"
                    
                    st.session_state.process_status["chunking"] = "running"
                    st.session_state.process_timings["chunking"] = "Auto"
                    st.session_state.process_status["chunking"] = "completed"
                    
                    st.session_state.process_status["embedding"] = "running"
                    st.session_state.process_timings["embedding"] = "Auto"
                    st.session_state.process_status["embedding"] = "completed"
                    
                    st.session_state.process_status["storage"] = "running"
                    st.session_state.process_timings["storage"] = "Auto"
                    st.session_state.process_status["storage"] = "completed"
                    
                    
                    
                    st.session_state.api_results = result
                    st.success("✅ Fast pipeline completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")

    elif st.session_state.current_mode == "config1":
        st.markdown("### ⚙️ Config-1 Mode Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧹 Preprocessing")
            null_handling = st.selectbox(
                "Null value handling",
                ["keep", "drop", "fill"],
                key="config1_null"
            )
            fill_value = st.text_input("Fill value", "Unknown") if null_handling == "fill" else None
            
            st.markdown("#### 📦 Chunking")
            chunk_method = st.selectbox(
                "Chunking method",
                ["fixed", "recursive", "semantic", "document"],
                key="config1_chunk"
            )
            
            if chunk_method in ["fixed", "recursive"]:
                chunk_size = st.number_input("Chunk size", 100, 2000, 400, key="config1_size")
                overlap = st.number_input("Overlap", 0, 500, 50, key="config1_overlap")
        
        with col2:
            st.markdown("#### 🤖 Embedding")
            model_choice = st.selectbox(
                "Embedding model",
                ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"],
                key="config1_model"
            )
            
            st.markdown("#### 💾 Storage")
            storage_choice = st.selectbox(
                "Vector storage",
                ["faiss", "chromadb"],
                key="config1_storage"
            )
            
            
        
        if st.button("🚀 Run Config-1 Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running Config-1 pipeline..."):
                try:
                    config = {
                        "null_handling": null_handling,
                        "fill_value": fill_value,
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 400,
                        "overlap": overlap if 'overlap' in locals() else 50,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                        
                    }
                    
                    # Update process status
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "running"
                    
                    result = call_config1_api(
                        st.session_state.uploaded_file.getvalue(),
                        st.session_state.uploaded_file.name,
                        config
                    )
                    
                    # Mark all as completed
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "completed"
                        st.session_state.process_timings[step] = "Completed"
                    
                    st.session_state.api_results = result
                    st.success("✅ Config-1 pipeline completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")

    elif st.session_state.current_mode == "deep":
        st.markdown("### 🔬 Deep Config Mode Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧹 Preprocessing")
            null_handling = st.selectbox(
                "Null value handling",
                ["keep", "drop", "fill"],
                key="deep_null"
            )
            fill_value = st.text_input("Fill value", "Unknown", key="deep_fill") if null_handling == "fill" else None
            
            st.markdown("#### 🧠 Text Processing")
            remove_stopwords = st.checkbox("Remove stopwords", key="deep_stop")
            lowercase = st.checkbox("Convert to lowercase", value=True, key="deep_lower")
            stemming = st.checkbox("Apply stemming", key="deep_stem")
            lemmatization = st.checkbox("Apply lemmatization", key="deep_lemma")
        
        with col2:
            st.markdown("#### 📦 Chunking")
            chunk_method = st.selectbox(
                "Chunking method",
                ["fixed", "recursive", "semantic", "document"],
                key="deep_chunk"
            )
            
            if chunk_method in ["fixed", "recursive"]:
                chunk_size = st.number_input("Chunk size", 100, 2000, 400, key="deep_size")
                overlap = st.number_input("Overlap", 0, 500, 50, key="deep_overlap")
            
            st.markdown("#### 🤖 Embedding & Storage")
            model_choice = st.selectbox(
                "Embedding model",
                ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"],
                key="deep_model"
            )
            storage_choice = st.selectbox(
                "Vector storage",
                ["faiss", "chromadb"],
                key="deep_storage"
            )
            
            
        
        if st.button("🚀 Run Deep Config Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running Deep Config pipeline..."):
                try:
                    config = {
                        "null_handling": null_handling,
                        "fill_value": fill_value,
                        "remove_stopwords": remove_stopwords,
                        "lowercase": lowercase,
                        "stemming": stemming,
                        "lemmatization": lemmatization,
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 400,
                        "overlap": overlap if 'overlap' in locals() else 50,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                        
                    }
                    
                    # Update process status
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "running"
                    
                    result = call_deep_api(
                        st.session_state.uploaded_file.getvalue(),
                        st.session_state.uploaded_file.name,
                        config
                    )
                    
                    # Mark all as completed
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "completed"
                        st.session_state.process_timings[step] = "Completed"
                    
                    st.session_state.api_results = result
                    st.success("✅ Deep Config pipeline completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")

    

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
    <p>📦 Chunking Optimizer • FastAPI + Streamlit • Vector DB Support</p>
</div>
""", unsafe_allow_html=True)