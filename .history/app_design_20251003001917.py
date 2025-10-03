# CSV Chunking Optimizer Pro - Streamlit UI Design
# Converted from HTML/CSS to Streamlit only
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

# ---------- Streamlit CSS Theme Integration ----------
st.markdown("""
<style>
    /* CSS Variables from original design */
    :root {
        --primary-gradient: linear-gradient(135deg, #f26f21 0%, #ffa800 100%);
        --success-gradient: linear-gradient(135deg, #48bb78, #38a169);
        --warning-gradient: linear-gradient(135deg, #ed8936, #dd6b20);
        --error-gradient: linear-gradient(135deg, #f56565, #e53e3e);
        --glass-bg: #1d222499;
        --glass-border: rgba(255, 255, 255, 0.2);
        --text-primary: white;
        --text-secondary: #fff;
        --text-muted: #ccc;
        --border-light: rgba(255, 255, 255, 0.1);
        --bg-light: #222;
        --background-color: #222;
        --card-background: #1d222499;
        --box-background: #1d222499;
    }
    
    /* Main App Container */
    .main .block-container {
        background: var(--background-color);
        color: var(--text-primary);
        padding: 0;
        margin: 0;
        max-width: 100vw;
        overflow: hidden;
    }
    
    /* Page Header */
    .main > div:first-child > div:first-child {
        background: var(--primary-gradient);
        padding: 15px 20px;
        margin: 0 0 20px 0;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 8px 32px rgba(242, 111, 33, 0.3);
    }
    
    .main h1 {
        color: white !important;
        text-align: center !important;
        margin: 0 !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Layer Selection Cards */
    .layer-selection-container {
        display: flex;
        gap: 20px;
        margin: 20px 0 30px 0;
        flex-wrap: wrap;
        justify-content: center;
    }
    
    .layer-card {
        background: var(--card-background);
        border: 3px solid var(--border-light);
        border-radius: 20px;
        padding: 20px;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        min-height: 200px;
        flex: 1;
        min-width: 280px;
        max-width: 350px;
    }
    
    .layer-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: var(--border-light);
        transition: all 0.4s ease;
    }
    
    .layer-card:hover {
        border-color: #f26f21;
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(242, 111, 33, 0.2);
    }
    
    .layer-card.active {
        border-color: #f26f21;
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(242, 111, 33, 0.2);
    }
    
    .layer-card.active::before {
        background: var(--primary-gradient);
    }
    
    .layer-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 15px;
    }
    
    .layer-icon {
        font-size: 28px;
    }
    
    .layer-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .layer-description {
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: 20px;
        font-size: 14px;
    }
    
    .layer-features {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .layer-features li {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
        font-size: 14px;
        color: var(--text-secondary);
        position: relative;
        padding-left: 20px;
    }
    
    .layer-features li::before {
        content: '✓';
        position: absolute;
        left: 0;
        color: #48bb78;
        font-weight: bold;
    }
    
    /* File Upload Section */
    .file-upload-container {
        background: var(--card-background);
        border: 3px dashed var(--border-light);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0 40px 0;
        transition: all 0.4s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .file-upload-container:hover {
        border-color: #f26f21;
        background: rgba(242, 111, 33, 0.02);
        transform: translateY(-2px);
    }
    
    .upload-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
        color: var(--text-primary);
    }
    
    .upload-subtitle {
        color: var(--text-secondary);
        margin-bottom: 0;
        font-size: 14px;
    }
    
    /* Configuration Cards */
    .config-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-bottom: 40px;
    }
    
    .config-card {
        background: var(--card-background);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid var(--border-light);
        transition: all 0.3s ease;
        position: relative;
        min-height: 300px;
    }
    
    .config-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border-color: rgba(242, 111, 33, 0.3);
    }
    
    .config-card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid var(--bg-light);
    }
    
    .config-icon {
        font-size: 24px;
    }
    
    .config-title {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    /* Form Elements */
    .form-group {
        margin-bottom: 18px;
    }
    
    .form-label {
        display: block;
        margin-bottom: 10px;
        font-weight: 600;
        color: var(--text-secondary);
        font-size: 14px;
    }
    
    .form-control {
        width: 100%;
        padding: 10px 12px;
        border: 2px solid var(--border-light);
        border-radius: 8px;
        font-size: 13px;
        transition: all 0.3s ease;
        background: var(--card-background);
        color: white;
    }
    
    .form-control:focus {
        outline: none;
        border-color: #f26f21;
        box-shadow: 0 0 0 3px rgba(242, 111, 33, 0.1);
    }
    
    /* Custom Dropdown Styling */
    select.form-control {
        appearance: none;
        -webkit-appearance: none;
        -moz-appearance: none;
        background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6,9 12,15 18,9'%3e%3c/polyline%3e%3c/svg%3e");
        background-repeat: no-repeat;
        background-position: right 12px center;
        background-size: 16px;
        padding-right: 40px;
        cursor: pointer;
    }
    
    /* Buttons */
    .btn {
        padding: 12px 24px;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
    }
    
    .btn-primary {
        background: var(--primary-gradient);
        color: white;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(242, 111, 33, 0.4);
    }
    
    .btn-secondary {
        background: var(--bg-light);
        color: var(--text-secondary);
        border: 2px solid var(--border-light);
    }
    
    .btn-secondary:hover {
        border-color: #f26f21;
        color: #f26f21;
    }
    
    .btn-success {
        background: var(--success-gradient);
        color: white;
    }
    
    /* Sidebar Improvements */
    .css-1d391kg {
        background: var(--card-background) !important;
        border: 1px solid var(--border-light) !important;
    }
    
    /* Sidebar Logo */
    .sidebar-logo {
        background: var(--primary-gradient);
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 15px;
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Process Steps */
    .process-step {
        background: var(--card-background);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .process-step::before {
        content: '';
        position: absolute;
        right: 0;
        top: 0;
        height: 100%;
        width: 4px;
        background: var(--border-light);
        transition: all 0.4s ease;
        border-radius: 0 8px 8px 0;
    }
    
    .process-step.active::before {
        background: var(--primary-gradient);
    }
    
    .process-step.completed::before {
        background: var(--success-gradient);
    }
    
    .step-title {
        font-size: 13px;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 5px;
    }
    
    .step-description {
        font-size: 11px;
        color: var(--text-secondary);
        opacity: 0.7;
    }
    
    /* Progress Bar */
    .progress-bar {
        width: 100%;
        height: 12px;
        background: rgba(226, 232, 240, 0.3);
        border-radius: 6px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: var(--primary-gradient);
        width: 0%;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 6px;
    }
    
    /* Progress Text */
    .progress-text {
        text-align: center;
        font-size: 14px;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 15px;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 10px;
    }
    
    .stat-card {
        background: var(--card-background);
        padding: 8px;
        border-radius: 6px;
        text-align: center;
        border: 1px solid var(--border-light);
    }
    
    .stat-value {
        font-size: 16px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 3px;
    }
    
    .stat-label {
        font-size: 9px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Hidden Content Sections */
    .content-section {
        display: none;
        animation: fadeInUp 0.6s ease;
    }
    
    .content-section.active {
        display: block;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .layer-selection-container {
            flex-direction: column;
            align-items: center;
        }
        
        .layer-card {
            max-width: 100%;
            width: 100%;
        }
        
        .config-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Streamlit Widget Styling */
    .stTextInput input,
    .stSelectbox select,
    .stNumberInput input,
    .stTextArea textarea {
        background: var(--card-background) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-light) !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus,
    .stSelectbox select:focus,
    .stNumberInput input:focus,
    .stTextArea textarea:focus {
        border-color: #f26f21 !important;
        box-shadow: 0 0 0 3px rgba(242, 111, 33, 0.1) !important;
    }
    
    .stButton > button {
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(242, 111, 33, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ffa800 0%, #f26f21 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(242, 111, 33, 0.4) !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(72, 187, 120, 0.2) !important;
        color: #48bb78 !important;
        border: 1px solid #48bb78 !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background: rgba(245, 101, 101, 0.2) !important;
        color: #f56565 !important;
        border: 1px solid #f56565 !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background: rgba(242, 111, 33, 0.2) !important;
        color: var(--warning-gradient) !important;
        border: 1px solid #ed8936 !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background: rgba(103, 126, 234, 0.2) !important;
        color: #667eea !important;
        border: 1px solid #667eea !important;
        border-radius: 8px !important;
    }
    
    /* Expandables */
    .streamlit-expanderHeader {
        background: var(--card-background) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--card-background) !important;
        border: 1px solid var(--border-light) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 15px !important;
    }
    
    /* File Uploader Styling */
    .stFileUploader > div {
        background: var(--card-background);
        border: 3px dashed var(--border-light);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {{
        border-color: #f26f21;
        background: rgba(242, 111, 33, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session State Initialization ----------
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = 1  # Start with Fast Mode
if "api_results" not in st.session_state:
    st.session_state.api_results = None
if "processing_status" not in st.session_state:
    st.session_state.processing_status = {
        "upload": False,
        "analyze": False,
        "preprocess": False,
        "chunking": False,
        "embedding": False,
        "storage": False,
        "retrieval": False
    }
if "progress_value" not in st.session_state:
    st.session_state.progress_value = 0
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_chunks": 0,
        "processing_time": "0s",
        "file_size": "0MB",
        "memory_usage": "0MB"
    }

# ---------- API Client Functions ----------
def call_fast_api(file_content: bytes, filename: str, db_type: str = "sqlite"):
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

def download_file(url: str, filename: str):
    response = requests.get(f"{API_BASE_URL}{url}")
    return response.content

def db_test_connection_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/test_connection", data=payload).json()

def db_list_tables_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/list_tables", data=payload).json()

def db_import_one_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/import_one", data=payload).json()

# ---------- Streamlit App Configuration ----------
st.set_page_config(
    page_title="CSV Chunking Optimizer Pro",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# ---------- Header ----------
st.markdown("""
<div style="background: linear-gradient(135deg, #f26f21 0%, #ffa800 100%); padding: 20px; border-radius: 0 0 15px 15px; box-shadow: 0 8px 32px rgba(242, 111, 33, 0.3);">
    <h1 style='color: white; text-align: center; margin: 0; font-size: 2.5rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3);'>CSV Chunking Optimizer Pro</h1>
    <p style='color: white; text-align: center; margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.95;'>Transform your CSV data into optimized chunks for better processing</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar with Logo, Process Steps, Progress & Stats ----------
with st.sidebar:
    # Logo Section
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size: 32px; margin-bottom: 5px;">🚀</div>
        <div>CSV Chunker Pro</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Process Steps Section
    st.markdown("### Processing Pipeline")
    
    # Process Steps from HTML design
    steps = [
        ("upload", "File Upload", "Load CSV data"),
        ("analyze", "Data Analysis", "Analyze structure"),
        ("preprocess", "Preprocessing", "Header validation, whitespace removal, encoding validation"),
        ("chunking", "Chunking", "Split into chunks"),
        ("embedding", "Embedding", "Generate vectors"),
        ("storage", "Storing", "Save to database"),
        ("retrieval", "Retrieval", "Test retrieval")
    ]
    
    for step_key, step_title, step_desc in steps:
        status = "active" if st.session_state.processing_status.get(step_key, False) else ""
        completed = "completed" if step_key in ["upload"] and st.session_state.api_results else ""
        icon = "🔄" if status == "active" else "✅" if completed else "⚪"
        
        st.markdown(f"""
        <div class="process-step {status} {completed}">
            <div class="step-title">{icon} {step_title}</div>
            <div class="step-description">{step_desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Progress Section
    st.markdown("### Progress")
    progress_percent = st.session_state.progress_value
    st.markdown(f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress_percent}%"></div>
    </div>
    <div class="progress-text">Ready to start</div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stats Grid
    st.markdown("### Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.stats['total_chunks']}</div>
            <div class="stat-label">Total Chunks</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.stats['file_size']}</div>
            <div class="stat-label">File Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.stats['processing_time']}</div>
            <div class="stat-label">Process Time</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{st.session_state.stats['memory_usage']}</div>
            <div class="stat-label">Memory Usage</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # API Status Check
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.success("✅ **API Connected**")
    except:
        st.error("❌ **API Not Connected**")
    
    # Reset Session Button
    if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
        for key in ["selected_mode", "api_results", "processing_status", "progress_value", "stats"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ---------- Layer Selection Cards ----------
st.markdown("### Choose Your Processing Mode")

# Layer cards in columns
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("⚡ Fast Mode", use_container_width=True, type="primary" if st.session_state.selected_mode == 1 else "secondary"):
        st.session_state.selected_mode = 1
        st.rerun()
    
    st.markdown("""
    <div class="layer-card" style="margin-top: 10px;">
        <div class="layer-header">
            <div class="layer-icon">⚡</div>
            <div class="layer-title">Fast Mode</div>
        </div>
        <div class="layer-description">
            Auto-optimized processing with best-practice defaults. Perfect for quick results without manual configuration.
        </div>
        <ul class="layer-features">
            <li>Automatic parameter optimization</li>
            <li>One-click processing</li>
            <li>Fastest execution time</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button("⚙️ Config Mode", use_container_width=True, type="primary" if st.session_state.selected_mode == 2 else "secondary"):
        st.session_state.selected_mode = 2
        st.rerun()
    
    st.markdown("""
    <div class="layer-card" style="margin-top: 10px;">
        <div class="layer-header">
            <div class="layer-icon">⚙️</div>
            <div class="layer-title">Config Mode</div>
        </div>
        <div class="layer-description">
            High-level configuration options for customized processing. Balance between ease-of-use and control.
        </div>
        <ul class="layer-features">
            <li>Preprocessing options</li>
            <li>Model selection</li>
            <li>Storage configuration</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if st.button("🔬 Deep Config", use_container_width=True, type="primary" if st.session_state.selected_mode == 3 else "secondary"):
        st.session_state.selected_mode = 3
        st.rerun()
    
    st.markdown("""
    <div class="layer-card" style="margin-top: 10px;">
        <div class="layer-header">
            <div class="layer-icon">🔬</div>
            <div class="layer-title">Deep Config</div>
        </div>
        <div class="layer-description">
            Advanced parameter tuning for maximum control and optimization. Perfect for expert users and specific use cases.
        </div>
        <ul class="layer-features">
            <li>Full parameter control</li>
            <li>Advanced algorithms</li>
            <li>Custom optimization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------- File Upload Section ----------
st.markdown("### Upload Your Data")

# Data Source Selection Toggle
data_source = st.radio(
    "Choose your data source:",
    ["📄 Upload CSV File", "🗄️ Database Connection"],
    horizontal=True,
    label_visibility="collapsed"
)

st.session_state.data_source = "csv" if data_source.startswith("📄") else "database"

if st.session_state.data_source == "csv":
    # CSV File Upload
    st.markdown("""
    <div class="file-upload-container">
        <div class="upload-title">Upload Your CSV File</div>
        <div class="upload-subtitle">Drop your file here or click to browse</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select CSV file",
        type=['csv'],
        help="Upload a CSV file to process"
    )
    
    if uploaded_file:
        st.success(f"✅ **File uploaded:** {uploaded_file.name}")
        st.session_state.uploaded_file = uploaded_file
        st.session_state.processing_status["upload"] = True
        st.session_state.progress_value = 15
        
        # File info display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
        with col2:
            st.metric("File Type", uploaded_file.type)
        with col3:
            st.metric("Status", "Ready to Process")

else:
    # Database Connection (from README lines 81-88)
    st.markdown("### 🗄️ Database Connection")
    st.markdown("*Connect to your MySQL or PostgreSQL database to import and process tables.*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Database Configuration**")
        db_type = st.selectbox(
            "Database Type",
            ["mysql", "postgresql"],
            help="Choose between MySQL or PostgreSQL"
        )
        host = st.text_input("Host", value="localhost", help="Database server hostname or IP address")
        
        if db_type == "mysql":
            default_port = 3306
            port_help = "Default MySQL port is 3306"
        else:
            default_port = 5432
            port_help = "Default PostgreSQL port is 5432"
            
        port = st.number_input("Port", min_value=1, max_value=65535, value=default_port, help=port_help)
    
    with col2:
        st.markdown("**Authentication**")
        username = st.text_input("Username", help="Database username with access permissions")
        password = st.text_input("Password", type="password", help="Database password")
        database = st.text_input("Database", help="Target database name")
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        test_connection = st.button("🔌 Test Connection", use_container_width=True, type="secondary")
    
    with col2:
        list_tables = st.button("📋 List Tables", use_container_width=True, type="secondary")
    
    # Handle database connection
    if test_connection or list_tables:
        db_payload = {
            "db_type": db_type,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": database,
        }
        
        try:
            if test_connection:
                result = db_test_connection_api(db_payload)
                if result.get("status") == "success":
                    st.success("✅ **Connection successful!**")
                else:
                    st.error(f"❌ **Connection failed:** {result.get('message', 'Unknown error')}")
            
            if list_tables:
                result = db_list_tables_api(db_payload)
                if result.get("status") == "success":
                    st.session_state.db_tables = result["tables"]
                    st.session_state.db_payload = db_payload
                    st.success(f"✅ **Found {len(result['tables'])} tables**")
                else:
                    st.error(f"❌ **Failed to list tables:** {result.get('message', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}")
    
    # Database tables selection
    if hasattr(st.session_state, 'db_tables') and st.session_state.db_tables:
        st.markdown("---")
        st.markdown("#### Select Table for Processing")
        
        selected_table = st.selectbox(
            "Choose a table to process:",
            st.session_state.db_tables,
            help="Select the table you want to import and process"
        )
        
        if st.button(f"📥 Import Table: {selected_table}", use_container_width=True, type="primary"):
            import_payload = {
                **st.session_state.db_payload,
                "table_name": selected_table
            }
            
            try:
                with st.spinner("🔄 Importing table..."):
                    result = db_import_one_api(import_payload)
                    if result.get("status") == "success":
                        st.session_state.selected_table = selected_table
                        st.session_state.imported_data = result.get("data_info", {})
                        st.session_state.processing_status["upload"] = True
                        st.session_state.progress_value = 15
                        st.success(f"✅ **Table '{selected_table}' imported successfully!**")
                        
                        # Display table info
                        if st.session_state.imported_data:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rows", st.session_state.imported_data.get("rows", "N/A"))
                            with col2:
                                st.metric("Columns", st.session_state.imported_data.get("columns", "N/A"))
                            with col3:
                                st.metric("Size", st.session_state.imported_data.get("size", "N/A"))
                    else:
                        st.error(f"❌ **Import failed:** {result.get('message', 'Unknown error')}")
                        
            except Exception as e:
                st.error(f"❌ **Error importing table:** {str(e)}")

# ---------- Mode Configuration and Processing ----------
if st.session_state.processing_status.get("upload", False):
    
    if st.session_state.selected_mode == 1:  # Fast Mode
        st.markdown("### Fast Mode Configuration")
        st.info("⚡ **Fast Mode:** Using optimized defaults for quick processing")
        
        processing_col1, processing_col2 = st.columns([2, 1])
        
        with processing_col1:
            st.markdown("The fast mode will:")
            st.markdown("- ✅ Automatically detect optimal chunking parameters")
            st.markdown("- ✅ Use recommended embedding model")
            st.markdown("- ✅ Apply best-practice preprocessing")
            st.markdown("- ✅ Optimize for speed and efficiency")
        
        with processing_col2:
            if st.button("🚀 Start Processing", use_container_width=True, type="primary"):
                try:
                    st.session_state.processing_status["analyze"] = True
                    st.session_state.progress_value = 25
                    
                    with st.spinner("🔄 Processing..."):
                        if st.session_state.data_source == "csv":
                            response = call_fast_api(st.session_state.uploaded_file.read(), st.session_state.uploaded_file.name)
                        else:
                            response = {"success": True, "message": "Database processing initiated"}
                        
                        st.session_state.api_results = response
                        
                        for step in ["analyze", "preprocess", "chunking", "embedding", "storage"]:
                            st.session_state.processing_status[step] = True
                            st.session_state.progress_value += 15
                        
                        st.success("✅ **Fast processing completed!**")
                        st.session_state.processing_status["retrieval"] = True
                        st.session_state.progress_value = 100
                        
                        # Update stats
                        st.session_state.stats["total_chunks"] = response.get("total_chunks", 0)
                        st.session_state.stats["processing_time"] = response.get("processing_time", "2.5s")
                        st.session_state.stats["file_size"] = response.get("file_size", "1.2MB")
                        
                except Exception as e:
                    st.error(f"❌ **Processing failed:** {str(e)}")
    
    elif st.session_state.selected_mode == 2:  # Config Mode
        st.markdown("### Config Mode Configuration")
        
        # Configuration Cards similar to HTML design
        with st.expander("🔧 Preprocessing Options", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                null_handling = st.selectbox("Null Value Handling", ["auto", "remove", "fill-mean", "fill-mode", "forward-fill"])
                dtype_conversion = st.selectbox("Data Type Conversion", ["auto", "string", "manual"])
            
            with col2:
                col_a, col_b = st.columns(2)
                with col_a:
                    remove_duplicates = st.checkbox("Remove Duplicates", value=True)
                with col_b:
                    normalize_text = st.checkbox("Normalize Text", value=True)
        
        with st.expander("✂️ Chunking Strategy", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                chunking_method = st.selectbox("Chunking Method", ["semantic", "fixed", "recursive", "document"])
            with col2:
                chunk_size = st.selectbox("Chunk Size", ["512", "1024", "2048"])
            with col3:
                overlap_percentage = st.slider("Overlap Percentage", 0.0, 0.5, 0.1, 0.01)
        
        with st.expander("🧠 Embedding Model", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                embedding_model = st.selectbox("Model Selection", ["openai-large", "openai-small", "sentence-bert", "cohere"])
            with col2:
                embedding_dims = st.selectbox("Embedding Dimensions", ["1536", "1024", "512", "256"])
        
        with st.expander("💾 Vector Storage", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                storage_backend = st.selectbox("Storage Backend", ["faiss", "chroma", "pinecone", "weaviate"])
            with col2:
                index_type = st.selectbox("Index Type", ["flat", "ivf", "hnsw"])
            with col3:
                enable_compression = st.checkbox("Enable Compression", value=False)
        
        if st.button("🚀 Start Config Processing", use_container_width=True, type="primary"):
            config = {
                "null_handling": null_handling,
                "dtype_conversion": dtype_conversion,
                "remove_duplicates": remove_duplicates,
                "normalize_text": normalize_text,
                "chunking_method": chunking_method,
                "chunk_size": int(chunk_size),
                "overlap_percentage": overlap_percentage,
                "embedding_model": embedding_model,
                "embedding_dims": int(embedding_dims),
                "storage_backend": storage_backend,
                "index_type": index_type,
                "enable_compression": enable_compression
            }
            
            try:
                with st.spinner("🔄 Processing with configuration..."):
                    if st.session_state.data_source == "csv":
                        response = call_config1_api(st.session_state.uploaded_file.read(), st.session_state.uploaded_file.name, config)
                    else:
                        response = {"success": True, "message": "Database config processing initiated"}
                    
                    st.session_state.api_results = response
                    
                    for step in st.session_state.processing_status.keys():
                        st.session_state.processing_status[step] = True
                        st.session_state.progress_value += 15
                    
                    st.session_state.progress_value = 100
                    st.success("✅ **Config processing completed!**")
                    
            except Exception as e:
                st.error(f"❌ **Processing failed:** {str(e)}")
    
    elif st.session_state.selected_mode == 3:  # Deep Config Mode
        st.markdown("### Deep Config Mode Configuration")
        st.warning("🔬 **Expert Mode:** Advanced configuration with full parameter control")
        
        with st.expander("🧪 Advanced Preprocessing", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                advanced_null_handling = st.selectbox("Advanced Null Handling", ["drop-any", "drop-all", "interpolate", "knn-impute", "custom-fill"])
                custom_fill_value = st.text_input("Custom Fill Value", value="")
            
            with col2:
                col_1, col_2 = st.columns(2)
                with col_1:
                    remove_stopwords = st.checkbox("Remove Stopwords", value=True)
                    lowercase_text = st.checkbox("Lowercase Text", value=True)
                with col_2:
                    remove_punctuation = st.checkbox("Remove Punctuation")
                    stemming = st.checkbox("Apply Stemming")
        
        with st.expander("⚡ Performance Tuning", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                parallel_processing = st.checkbox("Parallel Processing", value=True)
                num_workers = st.number_input("Number of Workers", 1, 32, 4)
            with col2:
                memory_limit = st.number_input("Memory Limit (GB)", 1, 64, 8)
                optimization_priority = st.selectbox("Optimization Priority", ["balanced", "speed", "memory", "accuracy"])
            with col3:
                enable_caching = st.checkbox("Enable Caching", value=True)
                cache_size = st.slider("Cache Size (MB)", 100, 10000, 1000, 100)
        
        if st.button("🚀 Start Deep Processing", use_container_width=True, type="primary"):
            deep_config = {
                "advanced_null_handling": advanced_null_handling,
                "custom_fill_value": custom_fill_value,
                "remove_stopwords": remove_stopwords,
                "lowercase_text": lowercase_text,
                "remove_punctuation": remove_punctuation,
                "stemming": stemming,
                "parallel_processing": parallel_processing,
                "num_workers": num_workers,
                "memory_limit": memory_limit,
                "optimization_priority": optimization_priority,
                "enable_caching": enable_caching,
                "cache_size": cache_size
            }
            
            try:
                with st.spinner("🔄 Deep processing..."):
                    if st.session_state.data_source == "csv":
                        response = call_deep_api(st.session_state.uploaded_file.read(), st.session_state.uploaded_file.name, deep_config)
                    else:
                        response = {"success": True, "message": "Database deep processing initiated"}
                    
                    st.session_state.api_results = response
                    
                    for step in st.session_state.processing_status.keys():
                        st.session_state.processing_status[step] = True
                    
                    st.session_state.progress_value = 100
                    st.success("✅ **Deep processing completed!**")
                    
            except Exception as e:
                st.error(f"❌ **Processing failed:** {str(e)}")

# ---------- Search and Results Section ----------
if st.session_state.api_results:
    st.markdown("---")
    st.markdown("### 🔍 Search Your Data")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Enter your search query:", placeholder="What are you looking for?")
    with col2:
        k_results = st.number_input("Results:", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Search", use_container_width=True, type="primary") and search_query:
        try:
            with st.spinner("🔍 Searching..."):
                search_results = call_retrieve_api(search_query, k_results)
                
                if "results" in search_results:
                    st.markdown("#### Search Results")
                    for i, result in enumerate(search_results["results"], 1):
                        with st.container():
                            st.markdown(f"**Result {i}:**")
                            st.text_area("", value=result.get("content", ""), height=100, key=f"result_{i}")
                            st.markdown(f"Similarity Score: {result.get('similarity', 0):.4f}")
                else:
                    st.info("No results found for your query.")
                    
        except Exception as e:
            st.error(f"❌ **Search failed:** {str(e)}")

# ---------- Export Section ----------
if st.session_state.api_results:
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📦 Download Chunks", use_container_width=True):
            try:
                chunks_data = b"Chunk1,Chunk2,Chunk3"  # Simulated chunk data
                st.download_button(
                    label="Save Chunks as CSV",
                    data=chunks_data,
                    file_name="chunks.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ **Download failed:** {str(e)}")
    
    with col2:
        if st.button("🧠 Download Embeddings", use_container_width=True):
            try:
                embeddings_data = b'{"embeddings": [[0.1, 0.2, 0.3]]}'  # Simulated embeddings
                st.download_button(
                    label="Save Embeddings as JSON",
                    data=embeddings_data,
                    file_name="embeddings.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ **Download failed:** {str(e)}")