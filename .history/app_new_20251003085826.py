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

# ---------- Initialize Session State ----------
if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {
        "preprocessing": False,
        "chunking": False,
        "embedding": False,
        "storage": False,
        "completed": False
    }
if 'csv_preview' not in st.session_state:
    st.session_state.csv_preview = None
if 'db_preview' not in st.session_state:
    st.session_state.db_preview = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

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

def download_file(url: str, filename: str):
    response = requests.get(f"{API_BASE_URL}{url}")
    return response.content

def display_preview_tables(preview_data):
    """Display preview tables for null analysis and data types"""
    if not preview_data or "error" in preview_data:
        if preview_data and "error" in preview_data:
            st.error(f"**Preview Error:** {preview_data['error']}")
        else:
            st.warning("⚠️ **No preview data available.** Click 'Preview Data' or 'Preview Table Data' first.")
        return
    
    # Debug info
    with st.expander("🔍 Debug Preview Data", expanded=False):
        st.json(preview_data)
    
    # Display Data Type Preview
    if "dtype_preview" in preview_data and preview_data["dtype_preview"]:
        st.markdown("#### Data Type Analysis")
        dtype_data = preview_data["dtype_preview"]["data"][:5]  # First 5 columns
        
        if dtype_data:
            dtype_df = pd.DataFrame(dtype_data)
            if not dtype_df.empty:
                st.dataframe(dtype_df, use_container_width=True, height=200)
                st.caption(f"Showing data types for {len(dtype_data)} columns")
        else:
            st.info("No data types preview available.")
    
    # Display Null Analysis Preview  
    if "null_preview" in preview_data and preview_data["null_preview"]:
        st.markdown("#### Null Value Analysis")
        null_data = preview_data["null_preview"]["data"][:5]  # First 5 columns
        
        if null_data:
            null_df = pd.DataFrame(null_data)
            if not null_df.empty:
                st.dataframe(null_df, use_container_width=True, height=200)
                st.caption(f"Showing null analysis for {len(null_data)} columns")
        else:
            st.info("No null analysis preview available.")
    
    # Display Summary Info
    if "total_rows" in preview_data:
        st.info(f"📋 **Dataset Info:** {preview_data['total_rows']} rows, {preview_data['total_columns']} columns")

def create_column_configuration_ui(columns: list, config_type: str, key_prefix: str):
    """Create per-column configuration UI"""
    configs = {}
    
    st.markdown(f"#### ⚙️ {config_type} Configuration (Per Column)")
    
    # Show which columns we'll configure
    st.info(f"📋 **Configuring options for {len(columns)} columns:** {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
    
    for i, col in enumerate(columns[:10]):  # Limit to first 10 columns for UI performance
        # Use expander for each column to save space
        with st.expander(f"⚙️ Configure {col}", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if config_type == "Null Handling":
                    configs[col] = st.selectbox(
                        f"Null handling method",
                        ["skip", "drop", "fill"],
                        key=f"{key_prefix}_null_{col}",
                        help=f"Choose how to handle null values in {col}"
                    )
                elif config_type == "Data Type":
                    configs[col] = st.selectbox(
                        f"Target data type",
                        ["skip", "numeric", "datetime", "bool", "object"],
                        key=f"{key_prefix}_dtype_{col}",
                        help=f"Choose target data type for {col}"
                    )
            
            with col2:
                if config_type == "Null Handling" and configs[col] == "fill":
                    fill_method = st.selectbox(
                        "Fill method",
                        ["median", "mode", "mean", "custom"],
                        key=f"{key_prefix}_fill_method_{col}",
                        help=f"Choose how to calculate fill value for {col}"
                    )
                    configs[f"{col}_fill_method"] = fill_method
                    
                    if fill_method == "custom":
                        custom_val = st.text_input(
                            "Custom value",
                            value="Unknown",
                            key=f"{key_prefix}_custom_{col}",
                            help=f"Custom value for {col}"
                        )
                        configs[f"{col}_custom_value"] = custom_val
    
    return configs

def db_test_connection_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/test_connection", data=payload).json()

def db_list_tables_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/list_tables", data=payload).json()

def db_import_one_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/import_one", data=payload).json()

# New preview API functions
def preview_csv_data(file_content: bytes, filename: str):
    """Preview CSV data with debugging"""
    try:
        files = {"file": (filename, file_content, "text/csv")}
        response = requests.post(f"{API_BASE_URL}/preview/data", files=files, timeout=30)
        print(f"API Response Status: {response.status_code}")
        result = response.json()
        print(f"API Response Data: {result}")
        return result
    except Exception as e:
        print(f"Preview API Error: {e}")
        return {"error": str(e)}

def preview_db_data(db_type: str, host: str, port: int, username: str, password: str, database: str, table_name: str):
    """Preview database data with debugging"""
    try:
        data = {
            "db_type": db_type,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": database,
            "table_name": table_name
        }
        response = requests.post(f"{API_BASE_URL}/preview/data", data=data, timeout=30)
        print(f"DB API Response Status: {response.status_code}")
        result = response.json()
        print(f"DB API Response Data: {result}")
        return result
    except Exception as e:
        print(f"DB Preview API Error: {e}")
        return {"error": str(e)}

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
            st.session_state.data_source = "csv"
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
                
                # Store preview data for later use
                if st.button("🔍 Preview Data", key="preview_csv_data", type="secondary"):
                    with st.spinner("Loading preview..."):
                        try:
                            preview = preview_csv_data(uploaded_file.getvalue(), uploaded_file.name)
                            st.write("**Debug Preview Response:**")
                            st.json(preview)  # Show raw response for debugging
                            
                            if "error" not in preview:
                                st.session_state.csv_preview = preview
                                st.success("✅ **Preview loaded successfully!**")
                                st.rerun()
                            else:
                                st.error(f"❌ **Preview Error:** {preview['error']}")
                        except Exception as e:
                            st.error(f"❌ **Error:** {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
    
    # Database Connection Section
    if st.session_state.data_source == "database":
        st.markdown("""
        <div class="data-source-container">
            <h3>🗄️ Database Connection</h3>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">
                Connect to your MySQL or PostgreSQL database to import and process tables.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Database connection form following README lines 81-88
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
        
        # Connection action buttons (moved outside form)
        col1, col2, spacer = st.columns([2, 2, 4], gap="small")
        
        with col1:
            test_connection = st.button(
                "🔌 Test Connection",
                use_container_width=True,
                type="secondary",
                help="Test database connectivity"
            )
        
        with col2:
            list_tables = st.button(
                "📋 List Tables",
                use_container_width=True,
                type="secondary",
                help="Get available tables"
            )

        # Process button clicks
        if test_connection or list_tables:
            db_payload = {
                "db_type": db_type,
                "host": host,
                "port": port,
                "username": username,
                "password": password,
                "database": database,
            }
            
            # Test connection
            if test_connection:
                with st.spinner("Testing connection..."):
                    try:
                        result = db_test_connection_api(db_payload)
                        if result.get("status") == "success":
                            st.success("**Connection Successful!**")
                            st.session_state.db_credentials = db_payload
                        else:
                            st.error(f"**Connection Failed:** {result.get('message', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"**Connection Error:** {str(e)}")
            
            # List tables
            if list_tables:
                with st.spinner("Fetching tables..."):
                    try:
                        result = db_list_tables_api(db_payload)
                        if "error" in result:
                            st.error(f"**Error:** {result['error']}")
                        else:
                            tables = result.get("tables", [])
                            if tables:
                                st.success(f"**Found {len(tables)} tables:**")
                                st.session_state.db_tables = tables
                                
                                # Display tables
                                for i, table in enumerate(tables, 1):
                                    st.write(f"{i}. **{table}**")
                                
                                # Store table list for later processing
                                st.session_state.db_credentials = db_payload
                            else:
                                st.warning("**No tables found in the database**")
                    except Exception as e:
                        st.error(f"**Error:** {str(e)}")

        # Database table selection and processing section
        if hasattr(st.session_state, 'db_tables') and st.session_state.db_tables:
            st.markdown("---")
            st.markdown("#### Select Table for Processing")
            
            selected_table = st.selectbox(
                "Choose a table to process:",
                st.session_state.db_tables,
                help="Select the table you want to import and process",
                key="db_table_selector"
            )
            
            if selected_table:
                st.session_state.selected_table = selected_table
                st.info(f"📋 **Table Selected:** {selected_table}")
                
                # Add preview button for database table
                if st.button("🔍 Preview Table Data", key="preview_db_data", type="secondary"):
                    with st.spinner("Loading table preview..."):
                        try:
                            preview = preview_db_data(db_type, host, port, username, password, database, selected_table)
                            st.write("**Debug Database Preview Response:**")
                            st.json(preview)  # Show raw response for debugging
                            
                            if "error" not in preview:
                                st.session_state.db_preview = preview
                                st.success("✅ **Table preview loaded successfully!**")
                                st.rerun()
                            else:
                                st.error(f"**Preview Error:** {preview['error']}")
                        except Exception as e:
                            st.error(f"❌ **Error:** {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())
                
                # Mode-specific processing controls
                st.markdown("#### Processing Options")
                
                # Fast Mode processing
                if st.session_state.selected_mode == "fast":
                    if st.button("📥 Import & Process (Fast Mode)", key="db_fast_process", use_container_width=True):
                        with st.spinner("Processing table with Fast Mode..."):
                            try:
                                payload = {
                                    **st.session_state.db_credentials,
                                    "table_name": selected_table
                                }
                                result = db_import_one_api(payload)
                                
                                if "error" in result:
                                    st.error(f"❌ **Processing Error:** {result['error']}")
                                else:
                                    st.success("✅ **Table processed successfully!**")
                                    st.session_state.api_results = result
                                    # Update processing status
                                    st.session_state.processing_status.update({
                                        "preprocessing": True,
                                        "chunking": True,
                                        "embedding": True,
                                        "storage": True,
                                        "completed": True
                                    })
                                    
                                    # Display results summary
                                    if "summary" in result:
                                        summary = result["summary"]
                                        st.markdown("#### 📊 Processing Results")
                                        st.json(summary)
                            except Exception as e:
                                st.error(f"❌ **Error:** {str(e)}")
                
                # Config-1 Mode processing
                elif st.session_state.selected_mode == "config1":
                    with st.expander("⚙️ Config-1 Settings", expanded=True):
                        c1_col1, c1_col2 = st.columns(2)
                        with c1_col1:
                            c1_null_handling = st.selectbox("Null Handling", ["keep", "drop", "fill"], key="db_c1_null")
                            c1_chunk_method = st.selectbox("Chunk Method", ["fixed", "recursive", "semantic", "document"], key="db_c1_chunk")
                            c1_model_choice = st.selectbox("Model", ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"], key="db_c1_model")
                        with c1_col2:
                            c1_fill_value = st.text_input("Fill Value", "Unknown", key="db_c1_fill") if c1_null_handling == "fill" else "Unknown"
                            c1_chunk_size = st.number_input("Chunk Size", 100, 2000, 400, key="db_c1_size")
                            c1_overlap = st.number_input("Overlap", 0, 200, 50, key="db_c1_overlap")
                            c1_storage_choice = st.selectbox("Storage", ["faiss", "chromadb"], key="db_c1_storage")
                    
                    if st.button("📥 Import & Process (Config-1 Mode)", key="db_config1_process", use_container_width=True):
                        with st.spinner("Processing table with Config-1 Mode..."):
                            try:
                                payload = {
                                    **st.session_state.db_credentials,
                                    "table_name": selected_table,
                                    "null_handling": c1_null_handling,
                                    "fill_value": c1_fill_value or "Unknown",
                                    "chunk_method": c1_chunk_method,
                                    "chunk_size": c1_chunk_size,
                                    "overlap": c1_overlap,
                                    "model_choice": c1_model_choice,
                                    "storage_choice": c1_storage_choice,
                                }
                                response = requests.post(f"{API_BASE_URL}/db/run_config1", data=payload)
                                result = response.json()
                                
                                if "error" in result:
                                    st.error(f"❌ **Processing Error:** {result['error']}")
                                else:
                                    st.success("✅ **Table processed successfully!**")
                                    st.session_state.api_results = result
                                    # Update processing status
                                    st.session_state.processing_status.update({
                                        "preprocessing": True,
                                        "chunking": True,
                                        "embedding": True,
                                        "storage": True,
                                        "completed": True
                                    })
                            except Exception as e:
                                st.error(f"❌ **Error:** {str(e)}")
                
                # Deep Mode processing
                elif st.session_state.selected_mode == "deep":
                    with st.expander("🔬 Deep Mode Settings", expanded=True):
                        deep_col1, deep_col2 = st.columns(2)
                        with deep_col1:
                            deep_null_handling = st.selectbox("Null Handling", ["keep", "drop", "fill"], key="db_deep_null")
                            deep_chunk_method = st.selectbox("Chunk Method", ["fixed", "recursive", "semantic", "document"], key="db_deep_chunk")
                            deep_model_choice = st.selectbox("Model", ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "sentence-transformers/all-mpnet-base-v2"], key="db_deep_model")
                            deep_remove_stopwords = st.checkbox("Remove Stopwords", key="db_deep_stopwords")
                            deep_lowercase = st.checkbox("Lowercase", value=True, key="db_deep_lowercase")
                        with deep_col2:
                            deep_fill_value = st.text_input("Fill Value", "Unknown", key="db_deep_fill") if deep_null_handling == "fill" else "Unknown"
                            deep_chunk_size = st.number_input("Chunk Size", 100, 2000, 400, key="db_deep_size")
                            deep_overlap = st.number_input("Overlap", 0, 200, 50, key="db_deep_overlap")
                            deep_storage_choice = st.selectbox("Storage", ["faiss", "chromadb"], key="db_deep_storage")
                            deep_stemming = st.checkbox("Stemming", key="db_deep_stemming")
                            deep_lemmatization = st.checkbox("Lemmatization", key="db_deep_lemmatization")
                  
                    if st.button("📥 Import & Process (Deep Mode)", key="db_deep_process", use_container_width=True):
                        with st.spinner("Processing table with Deep Mode..."):
                            try:
                                payload = {
                                    **st.session_state.db_credentials,
                                    "table_name": selected_table,
                                    "null_handling": deep_null_handling,
                                    "fill_value": deep_fill_value or "Unknown",
                                    "remove_stopwords": deep_remove_stopwords,
                                    "lowercase": deep_lowercase,
                                    "stemming": deep_stemming,
                                    "lemmatization": deep_lemmatization,
                                    "chunk_method": deep_chunk_method,
                                    "chunk_size": deep_chunk_size,
                                    "overlap": deep_overlap,
                                    "model_choice": deep_model_choice,
                                    "storage_choice": deep_storage_choice,
                                }
                                response = requests.post(f"{API_BASE_URL}/db/run_deep", data=payload)
                                result = response.json()
                                
                                if "error" in result:
                                    st.error(f"❌ **Processing Error:** {result['error']}")
                                else:
                                    st.success("✅ **Table processed successfully!**")
                                    st.session_state.api_results = result
                                    # Update processing status
                                    st.session_state.processing_status.update({
                                        "preprocessing": True,
                                        "chunking": True,
                                        "embedding": True,
                                        "storage": True,
                                        "completed": True
                                    })
                            except Exception as e:
                                st.error(f"❌ **Error:** {str(e)}")

# ---------- CSV Processing Pipeline ----------
if st.session_state.selected_mode and st.session_state.data_source == "csv" and hasattr(st.session_state, 'uploaded_file') and st.session_state.uploaded_file is not None:
    st.markdown("---")
    st.markdown("## 🚀 Processing Pipeline")
    
    # Mode-specific configuration based on selected mode
    if st.session_state.selected_mode == "fast":
        st.markdown("### ⚡ Fast Mode Configuration")
        st.markdown("""
        <div class="mode-card">
            <div class="mode-description">
                **Auto-optimized Pipeline**: This mode automatically handles preprocessing, 
                uses semantic clustering for chunking, MiniLM embeddings, and FAISS storage 
                for optimal performance with minimal configuration.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Run Fast Pipeline", key="csv_fast_process", use_container_width=True):
            with st.spinner("Running Fast Mode pipeline..."):
                try:
                    result = call_fast_api(
                        st.session_state.uploaded_file.getvalue(),
                        st.session_state.uploaded_file.name
                    )
                    
                    if "error" in result:
                        st.error(f"❌ **Processing Error:** {result['error']}")
                    else:
                        st.success("✅ **Fast pipeline completed successfully!**")
                        st.session_state.api_results = result
                        # Update processing status
                        st.session_state.processing_status.update({
                            "preprocessing": True,
                            "chunking": True,
                            "embedding": True,
                            "storage": True,
                            "completed": True
                        })
                        
                        # Display enhanced results summary
                        if "summary" in result:
                            summary = result["summary"]
                            st.markdown("#### 📊 Processing Results")
                            
                            # Display metadata if available
                            if "metadata" in summary:
                                metadata = summary["metadata"]
                                
                                # Data Source Info
                                if "data_source" in metadata:
                                    data_source = metadata["data_source"]
                                    st.success(f"📂 **Source:** {data_source.get('source_type', 'unknown')} - {data_source.get('filename', data_source.get('table_name', 'data'))}")
                                
                                # Processing Stats
                                if "processing_stats" in metadata:
                                    stats = metadata["processing_stats"]
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("📊 Rows Processed", f"{stats.get('rows_processed', 0):,}")
                                    with col2:
                                        st.metric("🏷️ Columns", stats.get('columns_processed', 0))
                                    with col3:
                                        st.metric("⏱️ Processing Time", f"{stats.get('preprocessing_time', 0):.2f}s")
                                    with col4:
                                        st.metric("💾 Memory Usage", f"{stats.get('memory_usage_mb', 0):.1f} MB")
                                
                                # Data Quality Info
                                if "data_quality" in metadata:
                                    quality = metadata["data_quality"]
                                    st.markdown("#### 🔍 Data Quality")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.info(f"🚫 **Null Values:** {quality.get('total_null_percentage', 0):.1f}%")
                                    with col2:
                                        st.info(f"📋 **Duplicates:** {quality.get('duplicate_rows', 0)} rows")
                                    with col3:
                                        st.info(f"📝 **Text Columns:** {len(quality.get('text_columns', []))}")
                                
                                # Show preview data from backend
                                if "null_preview" in metadata:
                                    st.markdown("#### 📈 Null Analysis Results")
                                    preview_df = pd.DataFrame(metadata["null_preview"]["data"])
                                    st.dataframe(preview_df, use_container_width=True, height=200)
                            
                            # Original summary (chunking results)
                            st.markdown("#### ⚙️ Pipeline Summary")
                            pipeline_summary = {
                                "rows": summary.get('rows', 0),
                                "chunks": summary.get('chunks', 0),
                                "vector_storage": summary.get('stored', 'unknown'),
                                "retrieval_ready": summary.get('retrieval_ready', False)
                            }
                            st.json(pipeline_summary)
                            
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")
    
    elif st.session_state.selected_mode == "config1":
        st.markdown("### ⚙️ Config-1 Mode Configuration")
        
        # Show preview data if available - Enhanced detection
        preview_data = None
        
        # Debug session state
        st.write("**🔍 Debug Session State:**")
        st.write(f"Data source: {st.session_state.get('data_source', 'none')}")
        st.write(f"Has csv_preview: {hasattr(st.session_state, 'csv_preview')}")
        st.write(f"Has db_preview: {hasattr(st.session_state, 'db_preview')}")
        
        # Check for CSV preview data
        if st.session_state.data_source == "csv" and hasattr(st.session_state, 'csv_preview') and st.session_state.csv_preview:
            preview_data = st.session_state.csv_preview
            st.success("Found CSV preview data!")
        
        # Check for DB preview data  
        elif st.session_state.data_source == "database" and hasattr(st.session_state, 'db_preview') and st.session_state.db_preview:
            preview_data = st.session_state.db_preview
            st.success("Found DB preview data!")
        
        # Always show preview section with instructions
        st.markdown("#### Data Preview & Analysis")
        if preview_data and "error" not in preview_data:
            st.success(f"**Preview data found!** Source: {preview_data.get('source_type', 'unknown')}")
            display_preview_tables(preview_data)
            st.markdown("---")
        else:
            st.warning("**Preview Required:** Click 'Preview Data' or 'Preview Table Data' button first to see column analysis and configure preprocessing options.")
            if preview_data and "error" in preview_data:
                st.error(f"**Preview Error:** {preview_data['error']}")
            st.markdown("---")
        
        with st.expander("⚙️ Config-1 Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🧹 Enhanced Preprocessing")
                
                # Per-column null handling configuration
                if preview_data and "null_preview" in preview_data and preview_data["null_preview"]:
                    st.write("Configure null handling for each column:")
                    columns = [item["column_name"] for item in preview_data["null_preview"]["data"] if isinstance(item, dict) and "column_name" in item]
                    if columns:
                        null_configs = create_column_configuration_ui(columns, "Null Handling", "csv_config1")
                    else:
                        st.warning("⚠️ **No column information found** in preview data.")
                else:
                    st.info("🔍 **Preview data first** to see per-column null handling options")
                
                st.markdown("#### 📦 Chunking")
                chunk_method = st.selectbox(
                    "Chunking method",
                    ["fixed", "recursive", "semantic", "document"],
                    key="csv_config1_chunk_method",
                    help="Choose chunking strategy"
                )
                
                if chunk_method in ["fixed", "recursive"]:
                    chunk_size = st.number_input("Chunk size", 100, 2000, 400, key="csv_config1_chunk_size")
                    overlap = st.number_input("Overlap", 0, 500, 50, key="csv_config1_overlap")
            
            with col2:
                st.markdown("#### 🤖 Embedding")
                model_choice = st.selectbox(
                    "Embedding model",
                    ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"],
                    key="csv_config1_model_choice",
                    help="Choose embedding model")
                    
                st.markdown("#### 💾 Storage")
                storage_choice = st.selectbox(
                    "Vector storage",
                    ["faiss", "chromadb"],
                    key="csv_config1_storage_choice",
                    help="Choose vector storage backend")
        
        if st.button("🚀 Run Config-1 Pipeline", key="csv_config1_process", use_container_width=True):
            with st.spinner("Running Config-1 pipeline..."):
                try:
                    # Build enhanced configuration
                    config = {
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 400,
                        "overlap": overlap if 'overlap' in locals() else 50,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                    }
                    
                    # Add null handling configuration if available
                    if preview_data and "null_preview" in preview_data and 'null_configs' in locals():
                        config["null_handling_config"] = null_configs
                    
                    result = call_config1_api(
                        st.session_state.uploaded_file.getvalue(),
                        st.session_state.uploaded_file.name,
                        config
                    )
                    
                    if "error" in result:
                        st.error(f"❌ **Processing Error:** {result['error']}")
                    else:
                        st.success("✅ **Config-1 pipeline completed successfully!**")
                        st.session_state.api_results = result
                        # Update processing status
                        st.session_state.processing_status.update({
                            "preprocessing": True,
                            "chunking": True,
                            "embedding": True,
                            "storage": True,
                            "completed": True
                        })
                        
                        # Display enhanced results summary (same as Fast Mode)
                        if "summary" in result:
                            summary = result["summary"]
                            st.markdown("#### 📊 Configuration Results")
                            
                            # Display metadata if available  
                            if "metadata" in summary:
                                metadata = summary["metadata"]
                                
                                # Data Source Info
                                if "data_source" in metadata:
                                    data_source = metadata["data_source"]
                                    st.success(f"📂 **Source:** {data_source.get('source_type', 'unknown')} - {data_source.get('filename', data_source.get('table_name', 'data'))}")
                                
                                # Processing Stats
                                if "processing_stats" in metadata:
                                    stats = metadata["processing_stats"]
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("📊 Rows Processed", f"{stats.get('rows_processed', 0):,}")
                                    with col2:
                                        st.metric("🏷️ Columns", stats.get('columns_processed', 0))
                                    with col3:
                                        st.metric("⏱️ Processing Time", f"{stats.get('preprocessing_time', 0):.2f}s")
                                    with col4:
                                        st.metric("💾 Memory Usage", f"{stats.get('memory_usage_mb', 0):.1f} MB")
                                
                                # Data Quality Info
                                if "data_quality" in metadata:
                                    quality = metadata["data_quality"]
                                    st.markdown("#### 🔍 Data Quality")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.info(f"🚫 **Null Values:** {quality.get('total_null_percentage', 0):.1f}%")
                                    with col2:
                                        st.info(f"📋 **Duplicates:** {quality.get('duplicate_rows', 0)} rows")
                                    with col3:
                                        st.info(f"📝 **Text Columns:** {len(quality.get('text_columns', []))}")
                            
                            # Pipeline summary
                            st.markdown("#### ⚙️ Pipeline Summary")
                            pipeline_summary = {
                                "rows": summary.get('rows', 0),
                                "chunks": summary.get('chunks', 0),
                                "vector_storage": summary.get('stored', 'unknown'),
                                "retrieval_ready": summary.get('retrieval_ready', False)
                            }
                            st.json(pipeline_summary)
                        
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")
    
    elif st.session_state.selected_mode == "deep":
        st.markdown("### Deep Mode Configuration")
        
        # Show preview data if available - Enhanced detection
        preview_data = None
        
        # Debug session state
        st.write("**🔍 Debug Session State:**")
        st.write(f"Data source: {st.session_state.get('data_source', 'none')}")
        st.write(f"Has csv_preview: {hasattr(st.session_state, 'csv_preview')}")
        st.write(f"Has db_preview: {hasattr(st.session_state, 'db_preview')}")
        
        # Check for CSV preview data
        if st.session_state.data_source == "csv" and hasattr(st.session_state, 'csv_preview') and st.session_state.csv_preview:
            preview_data = st.session_state.csv_preview
            st.success("Found CSV preview data!")
        
        # Check for DB preview data  
        elif st.session_state.data_source == "database" and hasattr(st.session_state, 'db_preview') and st.session_state.db_preview:
            preview_data = st.session_state.db_preview
            st.success("Found DB preview data!")
        
        # Always show preview section with instructions
        st.markdown("#### Data Preview & Analysis")
        if preview_data and "error" not in preview_data:
            st.success(f"**Preview data found!** Source: {preview_data.get('source_type', 'unknown')}")
            display_preview_tables(preview_data)
            st.markdown("---")
        else:
            st.warning("**Preview Required:** Click 'Preview Data' or 'Preview Table Data' button first to see column analysis and configure preprocessing options.")
            if preview_data and "error" in preview_data:
                st.error(f"**Preview Error:** {preview_data['error']}")
            st.markdown("---")
        
        with st.expander("🔬 Deep Mode Settings", expanded=True):
            # Tab-based organization for Deep Mode
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Types", "🔍 Null Handling", "📋 Data Processing", "⚙️ Advanced"])
            
            with tab1:
                st.markdown("#### Data Type Conversion")
                if preview_data and "dtype_preview" in preview_data and preview_data["dtype_preview"]:
                    st.write("Convert data types for each column:")
                    columns = [item["column_name"] for item in preview_data["dtype_preview"]["data"] if isinstance(item, dict) and "column_name" in item]
                    if columns:
                        dtype_configs = create_column_configuration_ui(columns, "Data Type", "csv_deep")
                    else:
                        st.warning("⚠️ **No column information found** in preview data.")
                else:
                    st.info("🔍 **Preview data first** to see per-column data type options")
            
            with tab2:
                st.markdown("#### Null Value Handling")
                if preview_data and "null_preview" in preview_data and preview_data["null_preview"]:
                    st.write("Configure null handling for each column:")
                    columns = [item["column_name"] for item in preview_data["null_preview"]["data"] if isinstance(item, dict) and "column_name" in item]
                    if columns:
                        null_configs = create_column_configuration_ui(columns, "Null Handling", "csv_deep")
                    else:
                        st.warning("⚠️ **No column information found** in preview data.")
                else:
                    st.info("🔍 **Preview data first** to see per-column null handling options")
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🗃️ Duplicate Handling")
                    duplicate_method = st.selectbox(
                        "Handle duplicates",
                        ["skip", "remove"],
                        key="csv_deep_duplicate_method",
                        help="Choose how to handle duplicate rows"
                    )
                    
                with col2:
                    st.markdown("#### 🧠 Text Processing")
                    text_processing_enabled = st.checkbox("Enable text processing", key="csv_deep_text_enabled")
                    
                    if text_processing_enabled:
                        text_method = st.selectbox(
                            "Text normalization method",
                            ["skip", "stemming", "lemmatization"],
                            key="csv_deep_text_method",
                            help="Choose text processing method"
                        )
            
            with tab4:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📦 Chunking")
                    chunk_method = st.selectbox(
                        "Chunking method",
                        ["fixed", "recursive", "semantic", "document"],
                        key="csv_deep_chunk_method",
                        help="Advanced chunking strategy"
                    )
                    
                    if chunk_method in ["fixed", "recursive"]:
                        chunk_size = st.number_input("Chunk size", 100, 2000, 400, key="csv_deep_chunk_size")
                        overlap = st.number_input("Overlap", 0, 500, 50, key="csv_deep_overlap")
                
                with col2:
                    st.markdown("#### 🤖 Embedding & Storage")
                    model_choice = st.selectbox(
                        "Embedding model",
                        ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"],
                        key="csv_deep_model_choice"
                    )
                    storage_choice = st.selectbox(
                        "Vector storage",
                        ["faiss", "chromadb"],
                        key="csv_deep_storage_choice"
                    )
        
        if st.button("🚀 Run Deep Config Pipeline", key="csv_deep_process", use_container_width=True):
            with st.spinner("Running Deep Config pipeline..."):
                try:
                    # Build enhanced configuration
                    config = {
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 400,
                        "overlap": overlap if 'overlap' in locals() else 50,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                    }
                    
                    # Add configurations from tabs
                    if 'dtype_configs' in locals():
                        config["dtype_config"] = dtype_configs
                    if 'null_configs' in locals():
                        config["null_handling_config"] = null_configs
                    if 'duplicate_method' in locals():
                        config["duplicate_config"] = {"method": duplicate_method}
                    if 'text_processing_enabled' in locals() and text_processing_enabled:
                        text_config = {
                            "enabled": True,
                            "stemming": text_method == "stemming",
                            "lemmatization": text_method == "lemmatization"
                        }
                        config["text_config"] = text_config
                    
                    result = call_deep_api(
                        st.session_state.uploaded_file.getvalue(),
                        st.session_state.uploaded_file.name,
                        config
                    )
                    
                    if "error" in result:
                        st.error(f"❌ **Processing Error:** {result['error']}")
                    else:
                        st.success("✅ **Deep Config pipeline completed successfully!**")
                        st.session_state.api_results = result
                        # Update processing status
                        st.session_state.processing_status.update({
                            "preprocessing": True,
                            "chunking": True,
                            "embedding": True,
                            "storage": True,
                            "completed": True
                        })
                        
                        # Display enhanced results summary
                        if "summary" in result:
                            summary = result["summary"]
                            st.markdown("#### 📊 Deep Config Results")
                            
                            # Display metadata if available
                            if "metadata" in summary:
                                metadata = summary["metadata"]
                                
                                # Data Source Info
                                if "data_source" in metadata:
                                    data_source = metadata["data_source"]
                                    st.success(f"📂 **Source:** {data_source.get('source_type', 'unknown')} - {data_source.get('filename', data_source.get('table_name', 'data'))}")
                                
                                # Processing Stats
                                if "processing_stats" in metadata:
                                    stats = metadata["processing_stats"]
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("📊 Rows Processed", f"{stats.get('rows_processed', 0):,}")
                                    with col2:
                                        st.metric("🏷️ Columns", stats.get('columns_processed', 0))
                                    with col3:
                                        st.metric("⏱️ Processing Time", f"{stats.get('preprocessing_time', 0):.2f}s")
                                    with col4:
                                        st.metric("💾 Memory Usage", f"{stats.get('memory_usage_mb', 0):.1f} MB")
                                
                                # Data Quality Info
                                if "data_quality" in metadata:
                                    quality = metadata["data_quality"]
                                    st.markdown("#### 🔍 Data Quality")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.info(f"🚫 **Null Values:** {quality.get('total_null_percentage', 0):.1f}%")
                                    with col2:
                                        st.info(f"📋 **Duplicates:** {quality.get('duplicate_rows', 0)} rows")
                                    with col3:
                                        st.info(f"📝 **Text Columns:** {len(quality.get('text_columns', []))}")
                            
                            # Pipeline summary
                            st.markdown("#### ⚙️ Pipeline Summary")
                            pipeline_summary = {
                                "rows": summary.get('rows', 0),
                                "chunks": summary.get('chunks', 0),
                                "vector_storage": summary.get('stored', 'unknown'),
                                "retrieval_ready": summary.get('retrieval_ready', False)
                            }
                            st.json(pipeline_summary)
                        
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")

# ---------- Vector Retrieval Section ----------
if hasattr(st.session_state, 'api_results') and st.session_state.api_results and st.session_state.api_results.get('summary', {}).get('retrieval_ready'):
    st.markdown("---")
    st.markdown("## 🔍 Semantic Search")
    st.markdown("Search for similar content using semantic similarity")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        vector_query = st.text_input("Enter semantic search query:", placeholder="Search for similar content...", key="vector_query")
    with col2:
        k = st.slider("Top K results", 1, 10, 3, key="vector_k")
    
    if vector_query and st.button("🔍 Search", key="vector_search"):
        with st.spinner("Searching..."):
            try:
                retrieval_result = call_retrieve_api(vector_query, k)
                
                if "error" in retrieval_result:
                    st.error(f"❌ **Retrieval error:** {retrieval_result['error']}")
                else:
                    st.success(f"✅ **Found {len(retrieval_result['results'])} results**")
                    
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
                st.error(f"❌ **Error:** {str(e)}")

# ---------- Export Section ----------
if hasattr(st.session_state, 'api_results') and st.session_state.api_results:
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
                st.error(f"❌ **Error exporting chunks:** {str(e)}")
    
    with col2:
        st.markdown("#### 📥 Download Embeddings")
        if st.button("🔢 Download Embeddings as NPY", use_container_width=True):
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
                st.error(f"❌ **Error exporting embeddings:** {str(e)}")


# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); font-size: 0.9em; opacity: 0.7;">
    <p>📦 Chunking Optimizer • Responsive Dark Theme • Advanced Vector Search</p>
</div>
""", unsafe_allow_html=True)
