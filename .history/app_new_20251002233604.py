# Responsive Streamlit UI Design for Chunking Optimizer
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

# ---------- Custom Dark Theme with Brand Colors ----------
st.markdown("""
<style>
    /* Custom Dark Theme Variables */
    :root {
        --text-secondary: #fff;
        --text-primary: white;
        --background: #121212;
        --brand-yellow: #ffa800;
        --orange: #f26f21;
        --card-outline: #313030;
        --section-outline-bottom: #222;
        --card-background: #1d222499;
        --text-disabled: #ffffff73;
        --badge-outline: #fff3;
        --button-press-outline: #919191e6;
        --button-press-bg: #0003;
        --article-text-color: white;
        --red: #d00000;
        --green: #1f5031;
        --brand-orange: #f26f21;
        --hero-subtitle-text: var(--text-primary);
    }
    
    /* Main App Background */
    .main .block-container {
        background: var(--background);
        color: var(--text-primary);
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, var(--brand-orange) 0%, var(--brand-yellow) 100%);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(242, 111, 33, 0.3);
    }
    
    .main-header h1 {
        color: var(--text-primary) !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: var(--text-secondary) !important;
        font-size: 1.2rem !important;
        margin: 0.5rem 0 0 0 !important;
        opacity: 0.95;
    }
    
    /* Mode Selection Cards */
    .mode-card {
        background: var(--card-background);
        border: 2px solid var(--card-outline);
        border-radius: 15px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
        height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }
    
    .mode-card:hover {
        border-color: var(--brand-orange);
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(242, 111, 33, 0.2);
    }
    
    .mode-card.selected {
        border-color: var(--brand-orange);
        background: rgba(242, 111, 33, 0.1);
        box-shadow: 0 8px 16px rgba(242, 111, 33, 0.3);
    }
    
    .mode-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--brand-orange), var(--brand-yellow));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .mode-card:hover::before,
    .mode-card.selected::before {
        opacity: 1;
    }
    
    .mode-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .mode-title {
        color: var(--text-primary) !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        text-align: center;
    }
    
    .mode-description {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        line-height: 1.4;
        margin-bottom: 1rem;
    }
    
    .mode-features {
        list-style: none;
        padding: 0;
        margin: 0;
        color: var(--text-secondary);
        font-size: 0.85rem;
    }
    
    .mode-features li {
        margin-bottom: 0.3rem;
        padding-left: 1rem;
        position: relative;
    }
    
    .mode-features li::before {
        content: '•';
        color: var(--brand-orange);
        font-weight: bold;
        position: absolute;
        left: 0;
    }
    
    /* Data Source Selection */
    .data-source-container {
        background: var(--card-background);
        border: 1px solid var(--card-outline);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .source-toggle {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .source-option {
        flex: 1;
        background: var(--card-background);
        border: 2px solid var(--card-outline);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .source-option:hover {
        border-color: var(--brand-orange);
        transform: translateY(-2px);
    }
    
    .source-option.selected {
        border-color: var(--brand-orange);
        background: rgba(242, 111, 33, 0.1);
    }
    
    .source-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .source-title {
        color: var(--text-primary) !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    
    .source-subtitle {
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        opacity: 0.8;
    }
    
    /* Database Connection Form */
    .db-form-container {
        background: var(--card-background);
        border: 1px solid var(--card-outline);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    .db-form-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .db-form-group {
        display: flex;
        flex-direction: column;
    }
    
    .db-form-group label {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* File Upload Styling */
    .stFileUploader > div {
        background: var(--card-background);
        border: 2px dashed var(--card-outline);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--brand-orange);
        background: rgba(242, 111, 33, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, var(--brand-orange), var(--brand-yellow)) !important;
        color: var(--text-primary) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(242, 111, 33, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, var(--brand-yellow), var(--brand-orange)) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(242, 111, 33, 0.4) !important;
    }
    
    /* Secondary Buttons */
    .secondary-button {
        background: var(--card-background) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--brand-orange) !important;
    }
    
    .secondary-button:hover {
        background: rgba(242, 111, 33, 0.1) !important;
        border-color: var(--brand-yellow) !important;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .mode-card {
            margin-bottom: 1rem;
            height: auto;
            min-height: 200px;
        }
        
        .source-toggle {
            flex-direction: column;
        }
        
        .db-form-grid {
            grid-template-columns: 1fr;
        }
        
        .main-header h1 {
            font-size: 2rem !important;
        }
        
        .main-header p {
            font-size: 1rem !important;
        }
    }
    
    @media (max-width: 480px) {
        .main-header {
            padding: 1.5rem;
        }
        
        .mode-card {
            padding: 1rem;
            height: auto;
        }
        
        .mode-icon {
            font-size: 2.5rem;
        }
        
        .source-icon {
            font-size: 2rem;
        }
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: var(--background) !important;
    }
    
    .css-1d391kg .css-17eq0hr {
        background: var(--card-background) !important;
        border: 1px solid var(--card-outline) !important;
        border-radius: 12px !important;
        margin-bottom: 1rem !important;
    }
    
    /* Form Elements */
    .stTextInput input,
    .stSelectbox select,
    .stNumberInput input {
        background: var(--background) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--card-outline) !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus,
    .stSelectbox select:focus,
    .stNumberInput input:focus {
        border-color: var(--brand-orange) !important;
        box-shadow: 0 0 0 2px rgba(242, 111, 33, 0.2) !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(31, 80, 49, 0.2) !important;
        color: var(--green) !important;
        border: 1px solid var(--green) !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background: rgba(208, 0, 0, 0.2) !important;
        color: var(--red) !important;
        border: 1px solid var(--red) !important;
        border-radius: 8px !important;
    }
    
    .stWarning {
        background: rgba(242, 111, 33, 0.2) !important;
        color: var(--brand-yellow) !important;
        border: 1px solid var(--brand-yellow) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session State Initialization ----------
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = None
if "data_source" not in st.session_state:
    st.session_state.data_source = "csv"
if "db_credentials" not in st.session_state:
    st.session_state.db_credentials = {}
if "processing_status" not in st.session_state:
    st.session_state.processing_status = {
        "preprocessing": False,
        "chunking": False,
        "embedding": False,
        "storage": False,
        "completed": False
    }

# ---------- Streamlit App Configuration ----------
st.set_page_config(
    page_title="Chunking Optimizer",
    layout="wide",
    page_icon="📦",
    initial_sidebar_state="expanded"
)

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

def db_test_connection_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/test_connection", data=payload).json()

def db_list_tables_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/list_tables", data=payload).json()

def db_import_one_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/import_one", data=payload).json()

# ---------- Main App Header ----------
st.markdown("""
<div class="main-header">
    <h1>📦 Chunking Optimizer</h1>
    <p>Advanced Text Processing + Vector Search Engine</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar: Process Tracker & System Info ----------
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(45deg, var(--brand-orange), var(--brand-yellow)); 
                padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem; text-align: center;">
        <h2 style="color: white; margin: 0; font-size: 1.3rem;">Process Tracker</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # API Status Check
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.success("✅ **API Connected**")
    except:
        st.error("❌ **API Not Connected**")
    
    st.markdown("---")
    
    # Processing Steps
    st.markdown("### ⚙️ Processing Steps")
    steps = [
        ("preprocessing", "🧹 Preprocessing"),
        ("chunking", "📦 Chunking"),
        ("embedding", "🤖 Embedding"),
        ("storage", "💾 Vector Storage"),
    ]
    
    for step_key, step_name in steps:
        status = st.session_state.processing_status.get(step_key, False)
        if status:
            st.markdown(f"✅ **{step_name}** - Completed")
        else:
            st.markdown(f"⚪ **{step_name}** - Pending")
    
    st.markdown("---")
    
    # Reset Session Button
    if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
        for key in ["selected_mode", "data_source", "db_credentials", "processing_status"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ---------- Three Processing Modes Selection ----------
st.markdown("## 🎯 Choose Processing Mode")

# Mode definitions
modes = [
    {
        "id": "fast",
        "icon": "⚡",
        "title": "Fast Mode",
        "description": "Auto-optimized pipeline for quick results",
        "features": [
            "Auto preprocessing",
            "Semantic clustering",
            "FAISS storage",
            "Quick results"
        ]
    },
    {
        "id": "config1", 
        "icon": "⚙️",
        "title": "Config-1 Mode",
        "description": "Customizable processing with balanced control",
        "features": [
            "Custom chunking",
            "Model selection",
            "Configurable settings",
            "Balanced performance"
        ]
    },
    {
        "id": "deep",
        "icon": "🔬", 
        "title": "Deep Mode",
        "description": "Advanced NLP with maximum control",
        "features": [
            "Advanced text processing",
            "All chunking methods",
            "NLP features",
            "Maximum quality"
        ]
    }
]

# Create responsive mode selection cards
cols = st.columns(3, gap="medium")

for i, mode in enumerate(modes):
    with cols[i]:
        selected_class = "selected" if st.session_state.selected_mode == mode["id"] else ""
        
        st.markdown(f"""
        <div class="mode-card {selected_class}" onclick="selectMode('{mode['id']}')">
            <div class="mode-icon">{mode['icon']}</div>
            <div class="mode-title">{mode['title']}</div>
            <div class="mode-description">{mode['description']}</div>
            <ul class="mode-features">
                {''.join([f'<li>{feature}</li>' for feature in mode['features']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Using st.button for interaction (JavaScript alternative)
        if st.button(f"Select {mode['title']}", key=f"mode_{mode['id']}", use_container_width=True):
            st.session_state.selected_mode = mode["id"]
            st.rerun()

# Show selected mode feedback
if st.session_state.selected_mode:
    selected_mode_obj = next(mode for mode in modes if mode["id"] == st.session_state.selected_mode)
    st.success(f"✅ **Selected: {selected_mode_obj['title']}** - {selected_mode_obj['description']}")

# ---------- Data Source Selection ----------
if st.session_state.selected_mode:
    st.markdown("---")
    st.markdown("## 📊 Data Source Selection")
    
    # Data source toggle
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        csv_selected = st.session_state.data_source == "csv"
        csv_class = "selected" if csv_selected else ""
        st.markdown(f"""
        <div class="source-option {csv_class}">
            <div class="source-icon">📄</div>
            <div class="source-title">CSV Upload</div>
            <div class="source-subtitle">Upload from your computer</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📄 Choose CSV Upload", key="source_csv", use_container_width=True):
            st.session_state.data_source = "数据库连接"
            st.rerun()
    
    with col2:
        db_selected = st.session_state.data_source == "database"
        db_class = "selected" if db_selected else ""
        st.markdown(f"""
        <div class="source-option {db_class}">
            <div class="source-icon">🗄️</div>
            <div class="source-title">Database Connection</div>
            <div class="source-subtitle">Connect to MySQL/PostgreSQL</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗄️ Choose Database", key="source_db", use_container_width=True):
            st.session_state.data_source = "database"
            st.rerun()
    
    # ---------- Data Source Specific Content ----------
    
    # CSV Upload Section
    if st.session_state.data_source == "csv":
        if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file is None:
            st.markdown("""
            <div class="data-source-container">
                <h3>📄 CSV File Upload</h3>
                <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
                    Upload your CSV file for processing. Supported formats: .csv files up to 200MB.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type=["csv"],
                help="Select a CSV file to process"
            )
            
            if uploaded_file is not None:
                st.session_state.uploaded_file = uploaded_file
                st.success(f"✅ **{uploaded_file.name}** uploaded successfully!")
                
                # File info display
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.info(f"📊 **File Size:** {file_size_mb:.2f} MB")
    
    # Database Connection Section
    if st.session_state.data_source == "database":
        st.markdown(f"""
        <div class="data-source-container">
            <h3>🗄️ Database Connection</h3>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
                Connect to your MySQL or PostgreSQL database to import and process tables.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Database connection form following README lines 81-88
        with st.form("database_connection_form"):
            st.markdown("#### Connection Settings")
            
            col1, col2 = st.columns(2, gap="medium")
            
            with col1:
                st.markdown("**Database Configuration**")
                db_type = st.selectbox(
                    "Select database type",
                    ["mysql", "postgresql"],
                    index=0,
                    help="Choose between MySQL or PostgreSQL"
                )
                
                host = st.text_input(
                    "Host",
                    value="localhost",
                    help="Database server hostname or IP address"
                )
                
                # Dynamic port based on database type
                if db_type == "mysql":
                    default_port = 3306
                    port_help = "Default MySQL port is 3306"
                else:  # postgresql
                    default_port = 5432
                    port_help = "Default PostgreSQL port is 5432"
                
                port = st.number_input(
                    "Port",
                    min_value=1,
                    max_value=65535,
                    value=default_port,
                    help=port_help
                )
            
            with col2:
                st.markdown("**Authentication**")
                
                username = st.text_input(
                    "Username",
                    help="Database username with access permissions"
                )
                
                password = st.text_input(
                    "Password",
                    type="password",
                    help="Database password"
                )
                
                database = st.text_input(
                    "Database",
                    help="Target database name"
                )
            
            st.markdown("---")
            
            # Connection action buttons
            col1, col2, spacer = st.columns([2, 2, 4], gap="small")
            
            with col1:
                test_connection = st.form_submit_button(
                    "🔌 Test Connection",
                    use_container_width=True,
                    type="secondary",
                    help="Test database connectivity"
                )
            
            with col2:
                list_tables = st.form_submit_button(
                    "📋 List Tables",
                    use_container_width=True,
                    type="secondary",
                    help="Get available tables"
                )
