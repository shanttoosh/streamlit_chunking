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
 
# Helper functions for deep config
def validate_and_normalize_headers(columns):
    """Validate and normalize column headers"""
    new_columns = []
    for i, col in enumerate(columns):
        if col is None or str(col).strip() == "":
            new_col = f"column_{i+1}"
        else:
            new_col = str(col).strip().lower()
        new_columns.append(new_col)
    return new_columns



# ---------- Minimalist Dark Theme ----------
st.markdown("""
<style>
    :root {
        --ev-colors-primary: #282828;
        --ev-colors-secondary: #424242;
        --ev-colors-tertiary: #4e332a;
        --ev-colors-highlight: #e75f33;
        --ev-colors-text: #fff;
        --ev-colors-secondaryText: grey;
        --ev-colors-tertiaryText: #a3a3a3;
        --ev-colors-borderColor: #ffffff1f;
        --ev-colors-background: #161616;
        --ev-colors-success: #d8fc77;
        --ev-colors-danger: #dc143c;
    }
    
    /* Main background */
    .stApp {
        background: var(--ev-colors-background);
        color: var(--ev-colors-text);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--ev-colors-text) !important;
        border-left: 4px solid var(--ev-colors-secondary) !important;
        padding-left: 10px !important;
    }
    
    /* Cards */
    .custom-card {
        background: var(--ev-colors-primary);
        border: 1px solid var(--ev-colors-borderColor);
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        background: var(--ev-colors-secondary);
    }
    
    .card-title {
        color: var(--ev-colors-text);
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    .card-content {
        color: var(--ev-colors-tertiaryText);
        font-size: 0.95em;
        line-height: 1.5;
    }
    
    /* Buttons - Only primary buttons use highlight color */
    .stButton > button {
        background: var(--ev-colors-secondary) !important;
        color: var(--ev-colors-text) !important;
        border: 1px solid var(--ev-colors-borderColor) !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: var(--ev-colors-tertiary) !important;
        border-color: var(--ev-colors-tertiaryText) !important;
    }
    
    /* Primary/Important buttons use highlight color */
    .primary-button > button {
        background: var(--ev-colors-highlight) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    .primary-button > button:hover {
        background: #f27024 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Process steps */
    .process-step {
        background: var(--ev-colors-primary);
        padding: 15px;
        border-radius: 6px;
        margin: 8px 0;
        border-left: 4px solid var(--ev-colors-secondary);
        transition: all 0.3s ease;
    }
    
    .process-step.running {
        border-left-color: var(--ev-colors-highlight);
    }
    
    .process-step.completed {
        border-left-color: var(--ev-colors-success);
    }
    
    .process-step.pending {
        border-left-color: var(--ev-colors-secondary);
    }
    
    /* Dataframes */
    .dataframe {
        background: var(--ev-colors-primary) !important;
        color: var(--ev-colors-text) !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background: var(--ev-colors-primary);
        color: var(--ev-colors-text);
        border: 1px solid var(--ev-colors-borderColor);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--ev-colors-highlight);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: var(--ev-colors-primary);
        color: var(--ev-colors-text);
        border: 1px solid var(--ev-colors-borderColor);
    }
    
    /* Number inputs */
    .stNumberInput > div > div > input {
        background: var(--ev-colors-primary);
        color: var(--ev-colors-text);
        border: 1px solid var(--ev-colors-borderColor);
    }
    
    /* Checkboxes & Radio buttons */
    .stCheckbox > label, .stRadio > label {
        color: var(--ev-colors-text) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--ev-colors-primary) !important;
    }
    
    /* Messages */
    .stSuccess {
        background: var(--ev-colors-primary) !important;
        color: var(--ev-colors-success) !important;
        border-left: 4px solid var(--ev-colors-success) !important;
    }
    
    .stError {
        background: var(--ev-colors-primary) !important;
        color: var(--ev-colors-danger) !important;
        border-left: 4px solid var(--ev-colors-danger) !important;
    }
    
    .stWarning {
        background: var(--ev-colors-primary) !important;
        color: var(--ev-colors-highlight) !important;
        border-left: 4px solid var(--ev-colors-highlight) !important;
    }
    
    .stInfo {
        background: var(--ev-colors-primary) !important;
        color: var(--ev-colors-text) !important;
        border-left: 4px solid var(--ev-colors-secondary) !important;
    }
    
    /* Text areas */
    .stTextArea > div > div > textarea {
        background: var(--ev-colors-primary);
        color: var(--ev-colors-text);
        border: 1px solid var(--ev-colors-borderColor);
    }
    
    /* Preview table */
    .preview-table {
        background: var(--ev-colors-primary);
        border: 1px solid var(--ev-colors-borderColor);
        border-radius: 6px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* File upload */
    .uploadedFile {
        background: var(--ev-colors-primary);
        border: 2px dashed var(--ev-colors-borderColor);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: var(--ev-colors-highlight);
    }
    
    /* Scrollable chunk display */
    .scrollable-chunk {
        background: var(--ev-colors-primary);
        border: 1px solid var(--ev-colors-borderColor);
        border-radius: 4px;
        padding: 10px;
        margin: 5px 0;
        max-height: 300px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.85em;
        line-height: 1.4;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .chunk-header {
        background: var(--ev-colors-secondary);
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-weight: bold;
        color: var(--ev-colors-text);
    }
    
    /* Scrollbar */
    .scrollable-chunk::-webkit-scrollbar {
        width: 6px;
    }
    
    .scrollable-chunk::-webkit-scrollbar-track {
        background: var(--ev-colors-primary);
    }
    
    .scrollable-chunk::-webkit-scrollbar-thumb {
        background: var(--ev-colors-secondary);
        border-radius: 3px;
    }
    
    .scrollable-chunk::-webkit-scrollbar-thumb:hover {
        background: var(--ev-colors-tertiaryText);
    }
    
    /* Minimal highlight usage */
    .highlight-text {
        color: var(--ev-colors-highlight);
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        color: var(--ev-colors-text);
        border-bottom: 1px solid var(--ev-colors-borderColor);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SVG Logo Integration ----------
logo_svg = """<svg id="Layer_2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1703.31 535.6"><defs><style>
      .cls-1 {
        fill: #fff;
      }

      .cls-2 {
        fill: #fbb03b;
      }

      .cls-3 {
        fill: #f27024;
      }
    </style></defs><g id="Layer_10"><g><path class="cls-1" d="M125.67,428.34c-39.15,0-70.27-13.09-92.48-38.91C11.17,363.84,0,334.47,0,302.15c0-30.4,9.47-57.88,28.14-81.68,23.77-30.39,56.01-45.8,95.83-45.8s74.1,15.76,98.58,46.85c17.39,21.95,26.36,49.63,26.66,82.28l.05,5.23H41.22c1.5,23.04,9.58,42.3,24.08,57.31,15.74,16.28,34.65,24.2,57.81,24.2,11.12,0,22.08-1.96,32.6-5.83,10.49-3.85,19.51-9.02,26.82-15.36,7.36-6.39,8.83-7.95,14.56-15.39l2.6-4.32c5.42-9.02,16.94-12.25,26.26-7.35h0c9.62,5.06,13.39,16.91,8.46,26.6l-1.53,3c-8.02,11.54-10.34,14.39-21.53,24.68-11.22,10.32-24.02,18.29-38.05,23.68-14.02,5.38-30.04,8.1-47.63,8.1ZM204.47,272.93c-3.65-12.13-8.55-22.08-14.6-29.64-7.06-8.82-16.57-16.06-28.27-21.51-11.75-5.46-24.27-8.23-37.2-8.23-21.29,0-39.83,6.92-55.1,20.58-9.88,8.81-17.76,21.84-23.46,38.8h158.64Z"></path><rect class="cls-1" x="288.28" y="97.26" width="40.15" height="331.08" rx="20.07" ry="20.07"></rect><path class="cls-1" d="M490.58,428.34c-39.15,0-70.27-13.09-92.48-38.91-22.02-25.59-33.19-54.96-33.19-87.28,0-30.4,9.47-57.88,28.14-81.68,23.77-30.39,56.01-45.8,95.83-45.8s74.1,15.76,98.58,46.85c17.39,21.95,26.36,49.63,26.66,82.28l.05,5.23h-208.03c1.5,23.04,9.58,42.3,24.08,57.31,15.74,16.28,34.65,24.2,57.81,24.2,11.12,0,22.08-1.96,32.6-5.83,10.49-3.85,19.51-9.02,26.82-15.36,7.36-6.39,8.83-7.95,14.56-15.39l2.6-4.32c5.42-9.02,16.94-12.25,26.26-7.35h0c9.62,5.06,13.39,16.91,8.46,26.6l-1.53,3c-8.02,11.54-10.34,14.39-21.53,24.68-11.22,10.32-24.02,18.29-38.05,23.68-14.02,5.38-30.04,8.1-47.63,8.1ZM569.37,272.93c-3.65-12.13-8.55-22.08-14.6-29.64-7.06-8.82-16.57-16.06-28.27-21.51-11.75-5.46-24.27-8.23-37.2-8.23-21.29,0-39.83,6.92-55.1,20.58-9.88,8.81-17.76,21.84-23.46,38.8h158.64Z"></path><path class="cls-1" d="M751.92,422.82l-96-208.47c-5.97-12.97,3.5-27.77,17.78-27.77h0c7.64,0,14.59,4.45,17.78,11.39l69.08,150.01,68.21-149.93c3.18-6.99,10.15-11.47,17.82-11.47h.22c14.26,0,23.74,14.76,17.8,27.73l-95.43,208.49c-1.55,3.38-4.92,5.54-8.63,5.54h0c-3.71,0-7.08-2.16-8.63-5.52Z"></path><g><path class="cls-2" d="M1052.79,311.55c-30.67,0-56.25,33.01-62.14,66.95,5.07-11.19,11.63-17.94,18.79-17.94,15.94,0,23.38,33.67,28.84,74.37,1.51,11.28,12.67,86.53,13.56,100.67.05,0,.11,0,.16,0,1.04-16.27,10.83-87.61,12.64-100.66,5.78-41.56,12.93-74.37,28.87-74.37,9.09,0,17.21,10.84,22.5,27.76-2.22-38.69-29.66-76.77-63.22-76.77Z"></path><path class="cls-3" d="M1053.33,46.78c60,50.38,96.73,131.67,97.74,218.86-26.55-32.52-60.86-50.27-97.76-50.27s-71.19,17.74-97.74,50.24c1.01-87.19,37.75-168.47,97.75-218.83M1053.33,0c-80.86,53.76-135.27,154.25-135.27,269.32,0,28.59,3.36,56.29,9.66,82.6,4.47,18.64,10.39,36.6,17.66,53.67,2.54-84.98,49.89-152.72,107.94-152.72s105.41,67.76,107.94,152.76c10.02-23.52,17.51-48.73,22.09-75.13,3.46-19.78,5.25-40.25,5.25-61.19C1188.59,154.25,1134.19,53.78,1053.33,0h0Z"></path></g><path class="cls-3" d="M1246.12,390.85l-15.96-370.06C1229.55,9.49,1238.55,0,1249.87,0h0c11.31,0,20.31,9.49,19.71,20.79l-15.96,370.06h-7.5Z"></path><path class="cls-1" d="M1333.96,408.27v-185.58h-40.62v-36.1h40.62v-69.25c0-11.09,8.99-20.07,20.07-20.07h0c11.09,0,20.07,8.99,20.07,20.07v69.25h62.21v36.1h-62.21v185.58c0,11.09-8.99,20.07-20.07,20.07h0c-11.09,0-20.07-8.99-20.07-20.07Z"></path><path class="cls-1" d="M1579.72,428.34c-39.15,0-70.26-13.09-92.48-38.91-22.02-25.59-33.18-54.95-33.18-87.28,0-30.4,9.47-57.88,28.14-81.68,23.77-30.39,56.01-45.8,95.83-45.8s74.1,15.76,98.59,46.85c17.39,21.94,26.36,49.63,26.66,82.28l.05,5.23h-208.03c1.5,23.04,9.59,42.3,24.08,57.31,15.74,16.28,34.64,24.2,57.81,24.2,11.12,0,22.09-1.96,32.6-5.83,10.49-3.85,19.51-9.02,26.82-15.36,7.36-6.39,9.22-7.53,15.54-17.02l1.62-2.69c5.42-9.02,16.94-12.25,26.26-7.35h0c9.62,5.06,13.39,16.91,8.46,26.6l-1.36,2.67c-6.09,8.44-10.51,14.72-21.7,25.01-11.22,10.32-24.02,18.29-38.06,23.68-14.02,5.38-30.04,8.1-47.63,8.1ZM1658.52,272.93c-3.65-12.13-8.55-22.08-14.6-29.64-7.06-8.82-16.57-16.06-28.27-21.51-11.76-5.46-24.27-8.23-37.2-8.23-21.29,0-39.83,6.92-55.1,20.58-9.89,8.81-17.76,21.85-23.46,38.8h158.64Z"></path></g></g></svg>"""

# Convert SVG to base64 and display
b64_logo = base64.b64encode(logo_svg.encode('utf-8')).decode("utf-8")

# Display logo and header
st.markdown(
    f'''
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="data:image/svg+xml;base64,{b64_logo}" width="300" alt="I Chunk Optimizer Logo">
    </div>
    <div style="background: var(--ev-colors-primary); border: 1px solid var(--ev-colors-borderColor); border-radius: 8px; padding: 20px; margin-bottom: 30px;">
        <h1 style="color: var(--ev-colors-text); text-align: center; margin: 0; font-size: 2.2em;">I Chunk Optimizer</h1>
        <p style="color: var(--ev-colors-tertiaryText); text-align: center; margin: 10px 0 0 0; font-size: 1.1em;">Advanced Text Processing + 3GB File Support + Performance Optimized</p>
    </div>
    ''',
    unsafe_allow_html=True
)

# ---------- API Client Functions ----------
def call_fast_api(file_path: str, filename: str, db_type: str, db_config: dict = None, 
                  use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                  process_large_files: bool = True, use_turbo: bool = False, batch_size: int = 256):
    """Send CSV upload or trigger DB import for Fast mode"""
    try:
        # DB import path: send only form data (no file open)
        if db_config and db_config.get('use_db'):
            data = {
                "db_type": db_config.get("db_type"),
                "host": db_config.get("host"),
                "port": db_config.get("port"),
                "username": db_config.get("username"),
                "password": db_config.get("password"),
                "database": db_config.get("database"),
                "table_name": db_config.get("table_name"),
                "use_openai": use_openai,
                "openai_api_key": openai_api_key,
                "openai_base_url": openai_base_url,
                "process_large_files": process_large_files,
                "use_turbo": use_turbo,
                "batch_size": batch_size
            }
            response = requests.post(f"{API_BASE_URL}/run_fast", data=data)
            return response.json()

        # CSV upload path: open and send file
        with open(file_path, 'rb') as f:
            files = {"file": (filename, f, "text/csv")}
            data = {
                "db_type": db_type,
                "use_openai": use_openai,
                "openai_api_key": openai_api_key,
                "openai_base_url": openai_base_url,
                "process_large_files": process_large_files,
                "use_turbo": use_turbo,
                "batch_size": batch_size
            }
            response = requests.post(f"{API_BASE_URL}/run_fast", files=files, data=data)
            return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

# Deep Config Step-by-Step API Functions
def call_deep_config_preprocess_api(file_path: str, filename: str, db_config: dict = None):
    """Step 1: Preprocess data"""
    try:
        if db_config and db_config.get('use_db'):
            data = {
                "db_type": db_config.get("db_type"),
                "host": db_config.get("host"),
                "port": db_config.get("port"),
                "username": db_config.get("username"),
                "password": db_config.get("password"),
                "database": db_config.get("database"),
                "table_name": db_config.get("table_name")
            }
            response = requests.post(f"{API_BASE_URL}/deep_config/preprocess", data=data)
        else:
            with open(file_path, 'rb') as f:
                files = {"file": (filename, f, "text/csv")}
                response = requests.post(f"{API_BASE_URL}/deep_config/preprocess", files=files)
        return response.json()
    except Exception as e:
        return {"error": f"Preprocess API call failed: {str(e)}"}

def call_deep_config_type_convert_api(type_conversions: dict):
    """Step 2: Convert data types"""
    try:
        data = {"type_conversions": json.dumps(type_conversions)}
        response = requests.post(f"{API_BASE_URL}/deep_config/type_convert", data=data)
        return response.json()
    except Exception as e:
        return {"error": f"Type convert API call failed: {str(e)}"}

def call_deep_config_null_handle_api(null_strategies: dict):
    """Step 3: Handle null values"""
    try:
        data = {"null_strategies": json.dumps(null_strategies)}
        response = requests.post(f"{API_BASE_URL}/deep_config/null_handle", data=data)
        return response.json()
    except Exception as e:
        return {"error": f"Null handle API call failed: {str(e)}"}

def call_deep_config_stopwords_api(remove_stopwords: bool):
    """Step 4: Remove stop words"""
    try:
        data = {"remove_stopwords": remove_stopwords}
        response = requests.post(f"{API_BASE_URL}/deep_config/stopwords", data=data)
        return response.json()
    except Exception as e:
        return {"error": f"Stopwords API call failed: {str(e)}"}

def call_deep_config_normalize_api(text_processing: str):
    """Step 5: Text normalization"""
    try:
        data = {"text_processing": text_processing}
        response = requests.post(f"{API_BASE_URL}/deep_config/normalize", data=data)
        return response.json()
    except Exception as e:
        return {"error": f"Normalize API call failed: {str(e)}"}

def call_deep_config_chunk_api(chunk_params: dict):
    """Step 6: Chunk data"""
    try:
        # Extract parameters from the dictionary
        chunk_method = chunk_params.get("method", "fixed")
        chunk_size = chunk_params.get("chunk_size", 400)
        overlap = chunk_params.get("overlap", 50)
        key_column = chunk_params.get("key_column")
        token_limit = chunk_params.get("token_limit", 2000)
        preserve_headers = chunk_params.get("preserve_headers", True)
        
        data = {
            "chunk_method": chunk_method,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "token_limit": token_limit,
            "preserve_headers": preserve_headers
        }
        if key_column:
            data["key_column"] = key_column
        response = requests.post(f"{API_BASE_URL}/deep_config/chunk", data=data)
        return response.json()
    except Exception as e:
        return {"error": f"Chunk API call failed: {str(e)}"}

def call_deep_config_embed_api(embed_params: dict):
    """Step 7: Generate embeddings"""
    try:
        # Extract parameters from the dictionary
        model_name = embed_params.get("model_name", "paraphrase-MiniLM-L6-v2")
        use_openai = embed_params.get("use_openai", False)
        openai_api_key = embed_params.get("openai_api_key")
        openai_base_url = embed_params.get("openai_base_url")
        batch_size = embed_params.get("batch_size", 64)
        use_parallel = embed_params.get("use_parallel", True)
        
        data = {
            "model_name": model_name,
            "use_openai": use_openai,
            "batch_size": batch_size
        }
        if openai_api_key:
            data["openai_api_key"] = openai_api_key
        if openai_base_url:
            data["openai_base_url"] = openai_base_url
        response = requests.post(f"{API_BASE_URL}/deep_config/embed", data=data)
        return response.json()
    except Exception as e:
        return {"error": f"Embed API call failed: {str(e)}"}

def call_deep_config_store_api(store_params: dict):
    """Step 8: Store embeddings"""
    try:
        # Extract parameters from the dictionary
        storage_type = store_params.get("storage_type", "chroma")
        collection_name = store_params.get("collection_name", "deep_config_collection")
        retrieval_metric = store_params.get("retrieval_metric", "cosine")
        
        data = {
            "storage_type": storage_type,
            "collection_name": collection_name
        }
        response = requests.post(f"{API_BASE_URL}/deep_config/store", data=data)
        return response.json()
    except Exception as e:
        return {"error": f"Store API call failed: {str(e)}"}

# Download functions for Deep Config
def download_deep_config_preprocessed():
    """Download preprocessed data"""
    response = requests.get(f"{API_BASE_URL}/deep_config/export/preprocessed")
    return response.content

def download_deep_config_chunks():
    """Download chunks"""
    response = requests.get(f"{API_BASE_URL}/deep_config/export/chunks")
    return response.content

def download_deep_config_embeddings():
    """Download embeddings"""
    response = requests.get(f"{API_BASE_URL}/deep_config/export/embeddings")
    return response.content


def call_config1_api(file_path: str, filename: str, config: dict, db_config: dict = None,
                    use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                    process_large_files: bool = True, use_turbo: bool = False, batch_size: int = 256):
    """Send CSV upload or trigger DB import for Config-1"""
    try:
        # DB import path: send only form data
        if db_config and db_config.get('use_db'):
            data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
            data.update({
                "db_type": db_config.get("db_type"),
                "host": db_config.get("host"),
                "port": db_config.get("port"),
                "username": db_config.get("username"),
                "password": db_config.get("password"),
                "database": db_config.get("database"),
                "table_name": db_config.get("table_name"),
                "use_openai": use_openai,
                "openai_api_key": openai_api_key,
                "openai_base_url": openai_base_url,
                "process_large_files": process_large_files,
                "use_turbo": use_turbo,
                "batch_size": batch_size
            })
            response = requests.post(f"{API_BASE_URL}/run_config1", data=data)
            return response.json()

        # CSV upload path: open and send file
        with open(file_path, 'rb') as f:
            files = {"file": (filename, f, "text/csv")}
            data = {k: str(v).lower() if isinstance(v, bool) else v for k, v in config.items()}
            data.update({
                "use_openai": use_openai,
                "openai_api_key": openai_api_key,
                "openai_base_url": openai_base_url,
                "process_large_files": process_large_files,
                "use_turbo": use_turbo,
                "batch_size": batch_size
            })
            response = requests.post(f"{API_BASE_URL}/run_config1", files=files, data=data)
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

def download_embeddings_text():
    """Download embeddings in text format"""
    response = requests.get(f"{API_BASE_URL}/export/embeddings_text")
    return response.content

# Database helper functions
def db_test_connection_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/test_connection", data=payload).json()

def db_list_tables_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/list_tables", data=payload).json()

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

# ---------- Scrollable Chunk Display Function ----------
def display_scrollable_chunk(result, chunk_index):
    """Display chunk content in a scrollable container"""
    similarity_color = "#28a745" if result['similarity'] > 0.7 else "#ffc107" if result['similarity'] > 0.4 else "#dc3545"
    
    # Create a unique key for the expander
    expander_key = f"chunk_{chunk_index}_{result['rank']}"
    
    with st.expander(f"📄 Rank #{result['rank']} (Similarity: {result['similarity']:.3f})", expanded=False):
        # Header with similarity score
        st.markdown(f"""
        <div style="background: #2d2d2d; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid {similarity_color};">
            <strong>Rank:</strong> {result['rank']} | 
            <strong>Similarity:</strong> {result['similarity']:.3f} | 
            <strong>Distance:</strong> {result.get('distance', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
        
        # Scrollable content area
        st.markdown("""
        <div class="chunk-header">
            📋 Chunk Content (Scrollable)
        </div>
        """, unsafe_allow_html=True)
        
        # Use text_area for scrollable content but make it read-only
        content = result['content']
        
        # Create a scrollable text area
        st.text_area(
            "Chunk Content",
            value=content,
            height=300,
            key=f"chunk_content_{chunk_index}",
            disabled=True,
            label_visibility="collapsed"
        )
        
        # Additional metadata if available
        if 'metadata' in result:
            st.markdown("""
            <div class="chunk-header">
                ℹ️ Metadata
            </div>
            """, unsafe_allow_html=True)
            st.json(result['metadata'])

# ---------- Streamlit App ----------
st.set_page_config(page_title="I Chunk Optimizer", layout="wide", page_icon="")

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
if "use_turbo" not in st.session_state:
    st.session_state.use_turbo = True
if "batch_size" not in st.session_state:
    st.session_state.batch_size = 256

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
            st.info("🚀 3GB+ File Support")
        if capabilities.get('performance_features', {}).get('turbo_mode'):
            st.info("⚡ Turbo Mode Available")
            
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
            st.info("""**Large File Features:**
            - Direct disk streaming (no memory overload)
            - Batch processing for memory efficiency
            - Automatic chunking for files >100MB
            - Progress tracking for large datasets
            - Support for 3GB+ files
            """)
    
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
        st.write(f"**Batch Size:** {system_info.get('embedding_batch_size', 'N/A')}")
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
        if file_info.get('turbo_mode'):
            st.success("⚡ Turbo Mode Enabled")
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
            if result['summary'].get('turbo_mode'):
                st.success("⚡ Turbo Mode Used")
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
    pass

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
                file_size_bytes = os.path.getsize(temp_path)
                
                # Check if file is large
                if is_large_file(file_size_bytes):
                    st.markdown(f"""
                    <div class="large-file-warning">
                        <strong>🚀 Large File Detected: {file_size_str}</strong><br>
                        Large file processing is {'ENABLED' if st.session_state.process_large_files else 'DISABLED'}<br>
                        <em>File streamed to disk - no memory overload</em>
                    </div>
                    """, unsafe_allow_html=True)
                
                # removed success banner per request
                
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
                if st.button("🔌 Test Connection", key="fast_test_conn", help="Tests DB connectivity. Then click “List Tables”."):
                    res = db_test_connection_api({
                        "db_type": db_type,
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "database": database,
                    })
                    if res.get("status") == "success":
                        import time as _t
                        st.session_state["fast_conn_ok_until"] = _t.time() + 5
                    else:
                        st.error(f"❌ Connection failed: {res.get('message', 'Unknown error')}")
                # Ephemeral success message under the button
                import time as _t
                if st.session_state.get("fast_conn_ok_until", 0) > _t.time():
                    st.markdown(
                        '<span style="padding:6px 10px; border:1px solid #444; border-radius:6px; background:#2d2d2d; color:#ddd;">Connection successful</span>',
                        unsafe_allow_html=True
                    )
            
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
        
        # FAST MODE DEFAULTS - No user configuration needed
        # Auto-enable turbo mode and set batch size to 256
        st.session_state.use_turbo = True
        st.session_state.batch_size = 256
        
        # Display Fast Mode pipeline with FIXED string formatting
        processing_type = "Parallel processing" if st.session_state.use_turbo else "Sequential processing"
        
        st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">Fast Mode Pipeline</div>
            <div class="card-content">
                • Optimized preprocessing for speed<br>
                • Semantic clustering chunking<br>
                • paraphrase-MiniLM-L6-v2 embedding model<br>
                • Batch embedding with size {st.session_state.batch_size}<br>
                • {processing_type}<br>
                • FAISS storage for fast retrieval<br>
                • 3GB+ file support with disk streaming<br>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
                            st.session_state.process_large_files,
                            st.session_state.use_turbo,
                            st.session_state.batch_size
                        )
                    else:
                        result = call_fast_api(
                            None, None, "sqlite", use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files,
                            st.session_state.use_turbo,
                            st.session_state.batch_size
                        )
                    
                    # Update process status
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "completed"
                        st.session_state.process_timings[step] = "Completed"
                    
                    st.session_state.api_results = result
                    # Show performance results
                    if 'summary' in result:
                        if result['summary'].get('large_file_processed'):
                            st.success("✅ Large file processed efficiently with disk streaming!")
                        elif result['summary'].get('turbo_mode'):
                            st.success("⚡ Turbo mode completed successfully!")
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
                file_size_bytes = os.path.getsize(temp_path)
                
                # Check if file is large
                if is_large_file(file_size_bytes):
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
                if st.button("🔌 Test Connection", key="config1_test_conn", help="Tests DB connectivity. Then click “List Tables”."):
                    res = db_test_connection_api({
                        "db_type": db_type,
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "database": database,
                    })
                    if res.get("status") == "success":
                        import time as _t
                        st.session_state["config1_conn_ok_until"] = _t.time() + 5
                    else:
                        st.error(f"❌ Connection failed: {res.get('message', 'Unknown error')}")
                # Ephemeral success message under the button
                import time as _t
                if st.session_state.get("config1_conn_ok_until", 0) > _t.time():
                    st.markdown(
                        '<span style="padding:6px 10px; border:1px solid #444; border-radius:6px; background:#2d2d2d; color:#ddd;">Connection successful</span>',
                        unsafe_allow_html=True
                    )
            
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
        
        # Config-1 parameters (refactored into tabs)
        st.markdown("#### ⚙️ Configuration Parameters")
        tab_chunk, tab_embed, tab_store = st.tabs(["Chunking", "Embedding", "Storage & Retrieval"])

        # Defaults to ensure variables exist for payload
        chunk_method = st.session_state.get("config1_chunk", "recursive")
        chunk_size = st.session_state.get("config1_size", 800)
        overlap = st.session_state.get("config1_overlap", 20)
        document_key_column = st.session_state.get("config1_document_key_column", "")
        token_limit = st.session_state.get("config1_token_limit", 2000)
        model_choice = st.session_state.get("config1_model", "paraphrase-MiniLM-L6-v2")
        storage_choice = st.session_state.get("config1_storage", "faiss")
        config1_retrieval_metric = st.session_state.get("config1_retrieval_metric", "cosine")

        with tab_chunk:
            st.markdown("#### 📦 Chunking")
            chunk_method = st.selectbox("Chunking method", ["fixed", "recursive", "semantic", "document"], key="config1_chunk")
            if chunk_method in ["fixed", "recursive"]:
                chunk_size = st.slider("Chunk size", 100, 2000, int(chunk_size), key="config1_size")
                overlap = st.slider("Overlap", 0, 500, int(overlap), key="config1_overlap")
            elif chunk_method == "document":
                st.markdown("#### 📄 Document Chunking Options")
                document_key_column = st.text_input(
                    "Key column (leave blank to use first column)",
                    key="config1_document_key_column",
                    value=str(document_key_column) if document_key_column else ""
                )
                token_limit = st.number_input(
                    "Token limit per chunk",
                    min_value=200,
                    max_value=10000,
                    value=int(token_limit),
                    step=100,
                    key="config1_token_limit"
                )
                # removed explanatory info text per request

        with tab_embed:
            st.markdown("#### 🤖 Embedding")
            model_choice = st.selectbox("Embedding model", 
                                      ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "text-embedding-ada-002"],
                                      key="config1_model")
            st.markdown("#### ⚡ Performance")
            st.session_state.use_turbo = st.checkbox(
                "Enable Turbo Mode", 
                value=st.session_state.use_turbo,
                help="Faster processing with parallel operations",
                key="config1_use_turbo"
            )
            st.session_state.batch_size = st.slider(
                "Embedding Batch Size",
                min_value=64,
                max_value=512,
                value=st.session_state.batch_size,
                step=64,
                help="Larger batches = faster processing (requires more memory)",
                key="config1_batch_size"
            )

        with tab_store:
            st.markdown("#### 💾 Storage")
            storage_choice = st.selectbox("Vector storage", ["faiss", "chromadb"], key="config1_storage", index=["faiss","chromadb"].index(storage_choice) if storage_choice in ["faiss","chromadb"] else 0)
            st.markdown("#### 🔎 Retrieval Metric")
            config1_retrieval_metric = st.selectbox(
                "Similarity metric",
                ["cosine", "dot", "euclidean"],
                index=["cosine","dot","euclidean"].index(config1_retrieval_metric) if config1_retrieval_metric in ["cosine","dot","euclidean"] else 0,
                key="config1_retrieval_metric"
            )
            # removed explanatory captions per request
        
        # removed turbo mode success banner per request
        
        run_enabled = (
            (input_source == "📁 Upload CSV File" and st.session_state.get('temp_file_path') is not None) or
            (input_source == "🗄️ Database Import" and use_db_config is not None)
        )
        
        if st.button("🚀 Run Config-1 Pipeline", type="primary", use_container_width=True, disabled=not run_enabled):
            with st.spinner("Running Config-1 pipeline..."):
                try:
                    config = {
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 800,
                        "overlap": overlap if 'overlap' in locals() else 20,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                    }
                    if chunk_method == "document":
                        if 'document_key_column' in locals() and document_key_column:
                            config["document_key_column"] = document_key_column
                        if 'token_limit' in locals() and token_limit:
                            config["token_limit"] = int(token_limit)
                    # include retrieval metric for storage compatibility
                    if 'config1_retrieval_metric' in locals() and config1_retrieval_metric:
                        config["retrieval_metric"] = config1_retrieval_metric
                    
                    if input_source == "📁 Upload CSV File":
                        result = call_config1_api(
                            st.session_state.temp_file_path,
                            st.session_state.file_info["name"],
                            config,
                            use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files,
                            st.session_state.use_turbo,
                            st.session_state.batch_size
                        )
                    else:
                        result = call_config1_api(
                            None, None, config, use_db_config,
                            st.session_state.use_openai,
                            st.session_state.openai_api_key,
                            st.session_state.openai_base_url,
                            st.session_state.process_large_files,
                            st.session_state.use_turbo,
                            st.session_state.batch_size
                        )
                    
                    # Mark all as completed
                    for step in ["preprocessing", "chunking", "embedding", "storage"]:
                        st.session_state.process_status[step] = "completed"
                        st.session_state.process_timings[step] = "Completed"
                    
                    st.session_state.api_results = result
                    # Show performance results
                    if 'summary' in result:
                        if result['summary'].get('large_file_processed'):
                            st.success("✅ Large file processed efficiently with disk streaming!")
                        elif result['summary'].get('turbo_mode'):
                            st.success("⚡ Turbo mode completed successfully!")
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
        st.markdown("### 🔬 Deep Config Mode - Comprehensive Workflow")
        
        # Import enhanced functions from backend
        try:
            from backend import (
                preprocess_csv_enhanced,
                profile_nulls_enhanced,
                suggest_null_strategy_enhanced,
                apply_null_strategies_enhanced,
                remove_stopwords_from_text_column_enhanced,
                process_text_enhanced,
                chunk_fixed_enhanced,
                chunk_semantic_cluster_enhanced,
                document_based_chunking_enhanced,
                chunk_recursive_keyvalue_enhanced,
                embed_texts_enhanced,
                store_chroma_enhanced,
                store_faiss_enhanced
            )
        except ImportError as e:
            st.error(f"Failed to import enhanced backend functions: {e}")
            st.error("Please ensure backend.py contains the enhanced functions")
            st.stop()
        
        # Initialize deep config session state variables
        if "deep_config_step" not in st.session_state:
            st.session_state.deep_config_step = 0
        if "preprocessing_config" not in st.session_state:
            st.session_state.preprocessing_config = {}
        if "chunking_config" not in st.session_state:
            st.session_state.chunking_config = {}
        if "embedding_config" not in st.session_state:
            st.session_state.embedding_config = {}
        if "storage_config" not in st.session_state:
            st.session_state.storage_config = {}
        if "deep_df" not in st.session_state:
            st.session_state.deep_df = pd.DataFrame()
        if "deep_file_meta" not in st.session_state:
            st.session_state.deep_file_meta = {}
        if "deep_numeric_meta" not in st.session_state:
            st.session_state.deep_numeric_meta = []
        if "deep_chunks" not in st.session_state:
            st.session_state.deep_chunks = []
        if "deep_chunking_result" not in st.session_state:
            st.session_state.deep_chunking_result = None
        if "deep_embedding_result" not in st.session_state:
            st.session_state.deep_embedding_result = None
        if "deep_meta_numeric_cols" not in st.session_state:
            st.session_state.deep_meta_numeric_cols = []
        if "deep_meta_categorical_cols" not in st.session_state:
            st.session_state.deep_meta_categorical_cols = []
        if "deep_store_metadata_enabled" not in st.session_state:
            st.session_state.deep_store_metadata_enabled = True

        # Input source selection for Deep Config
        input_source = st.radio("Select Input Source:", ["📁 Upload CSV File", "🗄️ Database Import"], key="deep_input_source")
        
        if input_source == "📁 Upload CSV File":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="deep_file_upload")
        else:  # Database Import
            st.markdown("#### 🗄️ Database Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                db_type = st.selectbox("Database Type", ["mysql", "postgresql"], key="deep_db_type")
                host = st.text_input("Host", "localhost", key="deep_host")
                port = st.number_input("Port", value=3306 if db_type == "mysql" else 5432, key="deep_port")
            
            with col2:
                username = st.text_input("Username", key="deep_username")
                password = st.text_input("Password", type="password", key="deep_password")
                database = st.text_input("Database Name", key="deep_database")
            
            # Test Connection and List Tables
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔌 Test Connection", key="deep_test_conn", help="Tests DB connectivity. Then click \"List Tables\"."):
                    res = db_test_connection_api({
                        "db_type": db_type,
                        "host": host,
                        "port": port,
                        "username": username,
                        "password": password,
                        "database": database,
                    })
                    if res.get("status") == "success":
                        import time as _t
                        st.session_state["deep_conn_ok_until"] = _t.time() + 5
                    else:
                        st.error(f"❌ Connection failed: {res.get('message', 'Unknown error')}")
                # Ephemeral success message under the button
                import time as _t
                if st.session_state.get("deep_conn_ok_until", 0) > _t.time():
                    st.markdown(
                        '<span style="padding:6px 10px; border:1px solid #444; border-radius:6px; background:#2d2d2d; color:#ddd;">Connection successful</span>',
                        unsafe_allow_html=True
                    )
            
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
                    if "error" in res:
                        st.error(f"❌ Failed to list tables: {res['error']}")
                    else:
                        st.session_state.deep_available_tables = res.get("tables", [])
                        st.success(f"✅ Found {len(st.session_state.deep_available_tables)} tables")
            
            # Table selection
            if hasattr(st.session_state, 'deep_available_tables') and st.session_state.deep_available_tables:
                table_name = st.selectbox("Select Table", st.session_state.deep_available_tables, key="deep_table_name")
                
                # Create DB config for Deep Config
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
                table_name = None
            
            uploaded_file = None  # No file upload for DB mode
        
        # Handle data loading for both CSV and DB
        if uploaded_file or (input_source == "🗄️ Database Import" and use_db_config):
            if st.session_state.deep_config_step == 0:
                if uploaded_file:
                    # CSV file upload
                    df = pd.read_csv(uploaded_file)
                    # Validate and normalize headers
                    df.columns = validate_and_normalize_headers(df.columns)
                    st.session_state.deep_df = df
                    st.session_state.deep_file_info = {
                        "source": "csv",
                        "filename": uploaded_file.name,
                        "size": len(uploaded_file.getvalue())
                    }
                else:
                    # DB import - load data via API
                    with st.spinner("🔄 Loading data from database..."):
                        st.session_state.deep_db_config = use_db_config
                        
                        # Load data directly from database for Deep Config UI preview
                        try:
                            from backend import connect_mysql, connect_postgresql, import_table_to_dataframe
                            
                            if use_db_config['db_type'] == 'mysql':
                                conn = connect_mysql(
                                    use_db_config['host'], 
                                    use_db_config['port'], 
                                    use_db_config['username'], 
                                    use_db_config['password'], 
                                    use_db_config['database']
                                )
                            elif use_db_config['db_type'] == 'postgresql':
                                conn = connect_postgresql(
                                    use_db_config['host'], 
                                    use_db_config['port'], 
                                    use_db_config['username'], 
                                    use_db_config['password'], 
                                    use_db_config['database']
                                )
                            
                            df = import_table_to_dataframe(conn, use_db_config['table_name'])
                            conn.close()
                            
                            # Validate and normalize headers
                            df.columns = validate_and_normalize_headers(df.columns)
                            st.session_state.deep_df = df
                            
                            st.session_state.deep_file_info = {
                                "source": f"db:{use_db_config['db_type']}",
                                "table": use_db_config['table_name'],
                                "database": use_db_config['database'],
                                "rows": len(df),
                                "columns": len(df.columns)
                            }
                            
                            st.success(f"✅ Successfully loaded {len(df)} rows from {use_db_config['table_name']}")
                            
                        except Exception as e:
                            st.error(f"❌ Failed to load database data: {str(e)}")
                            st.error(f"💡 **Troubleshooting**: Make sure you have the required database drivers installed:")
                            if use_db_config['db_type'] == 'postgresql':
                                st.error("   - For PostgreSQL: `pip install psycopg2-binary`")
                            elif use_db_config['db_type'] == 'mysql':
                                st.error("   - For MySQL: `pip install mysql-connector-python`")
                            st.session_state.deep_df = pd.DataFrame()

            st.subheader("Data preview")
            
            if not st.session_state.deep_df.empty:
                # Data summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", len(st.session_state.deep_df))
                with col2:
                    st.metric("Total Columns", len(st.session_state.deep_df.columns))
                with col3:
                    st.metric("Memory Usage", f"{st.session_state.deep_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                with col4:
                    null_count = st.session_state.deep_df.isnull().sum().sum()
                    st.metric("Null Values", null_count)
                
                # Enhanced scrollable dataframe
                st.subheader("Data Preview")
                st.dataframe(
                    st.session_state.deep_df,
                    height=300,
                    use_container_width=True,
                    hide_index=False
                )
            else:
                # Show DB import info
                if hasattr(st.session_state, 'deep_file_info') and st.session_state.deep_file_info.get('source', '').startswith('db:'):
                    st.info(f"📊 Database Import: {st.session_state.deep_file_info.get('table', 'Unknown table')} from {st.session_state.deep_file_info.get('database', 'Unknown database')}")
                    st.info("Data will be loaded during pipeline execution")

            # Step 1: Default preprocessing
            if st.session_state.deep_config_step == 0:
                if st.button("Run Default Preprocessing", key="deep_default_preprocessing"):
                    with st.spinner("🔄 Running preprocessing via API..."):
                        try:
                            # Determine input source and call API
                            if uploaded_file:
                                # CSV file upload
                                temp_path = None
                                try:
                                    # Create temporary file
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                                        tmp_file.write(uploaded_file.getvalue())
                                        temp_path = tmp_file.name
                                    
                                    result = call_deep_config_preprocess_api(temp_path, uploaded_file.name, None)
                                finally:
                                    # Clean up temp file
                                    if temp_path and os.path.exists(temp_path):
                                        os.unlink(temp_path)
                            else:
                                # DB import
                                result = call_deep_config_preprocess_api(None, None, use_db_config)
                            
                            if "error" in result:
                                st.error(f"❌ Preprocessing failed: {result['error']}")
                            else:
                                st.success(f"✅ Preprocessing completed successfully!")
                                st.info(f"📊 **Results**: {result.get('rows', 'N/A')} rows, {result.get('columns', 'N/A')} columns")
                                
                                # Update session state with API results
                                st.session_state.deep_file_meta = result.get('file_info', {})
                                st.session_state.deep_numeric_meta = []
                                st.session_state.deep_config_step = 1
                                st.rerun()
                        
                        except Exception as e:
                            st.error(f"❌ API Error: {str(e)}")

            # Step 2: Type conversion
            if st.session_state.deep_config_step == 1:
                st.sidebar.checkbox("Default Preprocessing Done", value=True, disabled=True, key="deep_step1_preprocessing_done")
                
                st.subheader("Data Type Conversion")
                # Back button
                if st.button("Back to Upload", key="deep_back_to_upload"):
                    st.session_state.deep_chunking_result = None
                    st.session_state.deep_embedding_result = None
                    st.session_state.deep_config_step = 0
                    st.rerun()
                
                # Check if data is available
                if st.session_state.deep_df.empty:
                    st.error("❌ No data available for type conversion. Please go back and load data first.")
                    if st.button("Back to Data Loading", key="deep_back_to_data_loading_step1"):
                        st.session_state.deep_config_step = 0
                        st.rerun()
                    st.stop()
                
                # Smart suggestions based on column names and data patterns
                def get_smart_suggestion(col_name, col_data):
                    col_name_lower = col_name.lower()
                    
                    if any(word in col_name_lower for word in ['date', 'time', 'created', 'updated', 'timestamp', 'birth', 'join']):
                        return 'datetime'
                    
                    if any(word in col_name_lower for word in ['flag', 'is_', 'has_', 'active', 'enabled', 'status', 'complaint']):
                        return 'boolean'
                    
                    if any(word in col_name_lower for word in ['count', 'score', 'price', 'amount', 'quantity', 'age', 'id']):
                        return 'float64'
                    
                    if col_data.dtype == 'object':
                        sample_values = col_data.dropna().head(10)
                        if len(sample_values) > 0:
                            date_patterns = [str(val) for val in sample_values if 
                                           any(char in str(val) for char in ['-', '/', ':']) and 
                                           len(str(val)) > 8]
                            if len(date_patterns) > len(sample_values) * 0.7:
                                return 'datetime'
                    
                    if col_data.dtype == 'object':
                        sample_values = col_data.dropna().head(10)
                        if len(sample_values) > 0:
                            bool_values = [str(val).lower() for val in sample_values if 
                                         str(val).lower() in ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n']]
                            if len(bool_values) > len(sample_values) * 0.7:
                                return 'boolean'
                    
                    if str(col_data.dtype) == 'int64':
                        return 'int64'
                    if str(col_data.dtype) == 'float64':
                        return 'float64'
                    return 'object'
                
                # Generate smart suggestions
                suggestions = {}
                for col in st.session_state.deep_df.columns:
                    suggestions[col] = get_smart_suggestion(col, st.session_state.deep_df[col])
                
                # Display smart suggestions overview
                st.write("🤖 **Smart Suggestions Overview:**")
                suggestion_df = pd.DataFrame({
                    'Column': st.session_state.deep_df.columns,
                    'Current Type': [str(dtype) for dtype in st.session_state.deep_df.dtypes],
                    'Suggested Type': [suggestions[col] for col in st.session_state.deep_df.columns],
                    'Reason': [
                        f"Detected {'date/time' if suggestions[col] == 'datetime' else 'boolean' if suggestions[col] == 'boolean' else 'numeric' if suggestions[col] in ['int64','float64'] else 'text'} patterns" 
                        for col in st.session_state.deep_df.columns
                    ]
                })
                
                st.dataframe(suggestion_df, use_container_width=True, height=250)
                
                # Group columns by their current data type
                dtype_groups = {}
                for col in st.session_state.deep_df.columns:
                    current_type = str(st.session_state.deep_df[col].dtype)
                    if current_type not in dtype_groups:
                        dtype_groups[current_type] = []
                    dtype_groups[current_type].append(col)
                
                # Initialize type conversions dictionary
                type_conversions = {}
                
                st.write("📋 **Grouped Conversion Interface:**")
                
                # Create conversion interface for each data type group
                for current_type, columns in dtype_groups.items():
                    st.write(f"**Current Type: `{current_type}` ({len(columns)} columns)**")
                    
                    # Get the most common suggestion for this group
                    group_suggestions = [suggestions[col] for col in columns]
                    most_common_suggestion = max(set(group_suggestions), key=group_suggestions.count)
                    
                    # Target conversion type selection
                    options = ["No change", "object", "int64", "float64", "datetime", "boolean"]
                    default_map = {
                        'object': 1,
                        'int64': 2,
                        'float64': 3,
                        'datetime': 4,
                        'boolean': 5,
                    }
                    target_type = st.selectbox(
                        f"Convert {current_type} columns to:",
                        options,
                        index=default_map.get(most_common_suggestion, 0),
                        key=f"deep_target_type_{current_type}"
                    )
                    
                    if target_type != "No change":
                        # Column selection checkboxes with smart suggestions
                        st.write("Select columns to convert:")
                        selected_columns = []
                        
                        # Create columns layout for checkboxes
                        cols_per_row = 3
                        for i in range(0, len(columns), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, col in enumerate(columns[i:i+cols_per_row]):
                                with cols[j]:
                                    suggested_type = suggestions[col]
                                    is_suggested = suggested_type == target_type
                                    if st.checkbox(f"{col} (suggested: {suggested_type})", 
                                                 value=is_suggested, 
                                                 key=f"deep_convert_{col}"):
                                        selected_columns.append(col)
                        
                        # Add selected columns to type conversions
                        for col in selected_columns:
                            type_conversions[col] = target_type
                        
                        # Show smart suggestion summary
                        suggested_cols = [col for col in columns if suggestions[col] == target_type]
                        if suggested_cols:
                            st.info(f"💡 Smart suggestion: {target_type} for {', '.join(suggested_cols)}")
                    
                    st.divider()
                
                # Apply type conversion button
                if st.button("Apply Type Conversion", key="deep_apply_type_conversion"):
                    if type_conversions:
                        with st.spinner(f"🔄 Converting {len(type_conversions)} columns via API..."):
                            try:
                                result = call_deep_config_type_convert_api(type_conversions)
                                
                                if "error" in result:
                                    st.error(f"❌ Type conversion failed: {result['error']}")
                                else:
                                    st.success(f"✅ Type conversion completed successfully!")
                                    st.info(f"📊 **Converted**: {len(type_conversions)} columns")
                                    
                                    # Update session state with API results
                                    st.session_state.deep_file_meta = result.get('file_info', {})
                                    st.session_state.deep_numeric_meta = []
                                    st.session_state.deep_config_step = 2
                                    st.rerun()
                            
                            except Exception as e:
                                st.error(f"❌ API Error: {str(e)}")
                    else:
                        st.info("No type conversions selected. Moving to next step.")
                        st.session_state.deep_config_step = 2
                        st.rerun()

                # Skip button
                if st.button("Apply No Changes (Skip Type Conversion)", key="deep_skip_type_conversion"):
                    st.info("Type conversion skipped.")
                    st.session_state.deep_config_step = 2
                    st.rerun()

            # Step 3: Null handling
            if st.session_state.deep_config_step == 2:
                st.sidebar.checkbox("Default Preprocessing Done", value=True, disabled=True, key="deep_step2_preprocessing_done")
                st.sidebar.checkbox("Type Conversion Done", value=True, disabled=True, key="deep_step2_type_conversion_done")
                st.subheader("Null Handling")
                # Back button
                if st.button("Back to Data Types", key="deep_back_to_types"):
                    st.session_state.deep_chunking_result = None
                    st.session_state.deep_embedding_result = None
                    st.session_state.deep_config_step = 1
                    st.rerun()
                
                # Check if data is available
                if st.session_state.deep_df.empty:
                    st.error("❌ No data available for null handling. Please go back and load data first.")
                    if st.button("Back to Data Loading", key="deep_back_to_data_loading"):
                        st.session_state.deep_config_step = 0
                        st.rerun()
                    st.stop()
                
                st.write("Smart suggestions based on null ratio, dtype, and column semantics.")

                # Smart suggestions overview
                profile = profile_nulls_enhanced(st.session_state.deep_df)
                profile['suggested'] = profile.apply(lambda r: suggest_null_strategy_enhanced(r['column'], st.session_state.deep_df[r['column']]), axis=1)
                filtered_profile = profile[profile['null_count'] > 0].copy()

                if filtered_profile.empty:
                    st.info("No null values detected; you can proceed to the next step.")
                    if st.button("Proceed to Stop Words Removal", key="deep_proceed_no_nulls"):
                        st.session_state.deep_config_step = 4
                        st.rerun()
                else:
                    st.write("🤖 Smart Suggestions (Nulls):")
                    st.dataframe(
                        filtered_profile[['column','dtype','null_count','null_pct','suggested']].sort_values('null_pct', ascending=False),
                        use_container_width=True,
                        height=250
                    )

                # Grouped controls by dtype
                st.write("📋 **Grouped Controls:**")
                type_groups = {}
                for _, row in filtered_profile.iterrows():
                    type_groups.setdefault(row['dtype'], []).append(row['column'])

                null_strategies = {}
                options = ["No change", "leave", "drop", "mean", "median", "mode", "zero", "unknown", "ffill", "bfill"]
                for dtype_name, cols in type_groups.items():
                    st.write(f"**{dtype_name}** ({len(cols)} columns)")
                    group_sugs = [filtered_profile.loc[filtered_profile['column']==c, 'suggested'].values[0] for c in cols]
                    if group_sugs:
                        from collections import Counter
                        most_common = Counter(group_sugs).most_common(1)[0][0]
                        default_index = options.index(most_common) if most_common in options else 0
                    else:
                        default_index = 0
                    group_choice = st.selectbox(
                        f"Default strategy for {dtype_name}:",
                        options,
                        index=default_index,
                        key=f"deep_null_group_{dtype_name}"
                    )

                    # Per-column overrides
                    cols_per_row = 3
                    for i in range(0, len(cols), cols_per_row):
                        ccols = st.columns(cols_per_row)
                        for j, col in enumerate(cols[i:i+cols_per_row]):
                            with ccols[j]:
                                sug = filtered_profile.loc[filtered_profile['column']==col, 'suggested'].values[0]
                                choice = st.selectbox(
                                    f"{col} (sugg: {sug})",
                                    options,
                                    index=options.index(sug) if sug in options else default_index,
                                    key=f"deep_null_choice_{col}"
                                )
                                if choice != "No change":
                                    null_strategies[col] = choice

                    st.divider()

                if st.button("Apply Null Handling", key="deep_apply_null_handling"):
                    with st.spinner("🔄 Applying null handling via API..."):
                        try:
                            result = call_deep_config_null_handle_api(null_strategies)
                            
                            if "error" in result:
                                st.error(f"❌ Null handling failed: {result['error']}")
                            else:
                                st.success(f"✅ Null handling completed successfully!")
                                st.info(f"📊 **Processed**: {len(null_strategies)} columns with null strategies")
                                
                                # Update session state with API results
                                st.session_state.deep_file_meta = result.get('file_info', {})
                                st.session_state.deep_numeric_meta = []
                                st.session_state.deep_config_step = 4
                                st.rerun()
                        
                        except Exception as e:
                            st.error(f"❌ API Error: {str(e)}")
                    
                # Skip button
                if st.button("Apply No Changes (Skip Null Handling)", key="deep_skip_null_handling"):
                    st.info("Null handling skipped.")
                    st.session_state.deep_config_step = 4
                    st.rerun()

            # Step 4: Stop Words Removal
            if st.session_state.deep_config_step == 4:
                st.subheader("Stop Words Removal")
                # Back button
                if st.button("Back to Null Handling", key="deep_back_to_nulls"):
                    st.session_state.deep_chunking_result = None
                    st.session_state.deep_embedding_result = None
                    st.session_state.deep_config_step = 2
                    st.rerun()

                # Check if data is available
                if st.session_state.deep_df.empty:
                    st.error("❌ No data available for stop words removal. Please go back and load data first.")
                    if st.button("Back to Data Loading", key="deep_back_to_data_loading_step4"):
                        st.session_state.deep_config_step = 0
                        st.rerun()
                    st.stop()

                text_cols = st.session_state.deep_df.select_dtypes(include=["object"]).columns
                if text_cols.empty:
                    st.info("No text column found to apply stop words removal.")
                    if st.button("Proceed to Normalization (Skip Stopwords)", key="deep_proceed_no_text_stopwords"):
                        st.session_state.deep_config_step = 5
                        st.rerun()
                else:
                    choice = st.radio(
                        "Choose stop word handling:",
                        ["Apply stop word removal", "Skip stop word removal"],
                        index=1,
                        key="deep_stopword_choice"
                    )

                    if st.button("Continue", key="deep_continue_stopwords"):
                        with st.spinner("🔄 Processing stop words via API..."):
                            try:
                                remove_stopwords = choice == "Apply stop word removal"
                                result = call_deep_config_stopwords_api(remove_stopwords)
                                
                                if "error" in result:
                                    st.error(f"❌ Stop words processing failed: {result['error']}")
                                else:
                                    if remove_stopwords:
                                        st.success("✅ Stop words removed from detected text columns.")
                                    else:
                                        st.info("Stop words removal skipped.")
                                    
                                    # Update session state with API results
                                    st.session_state.deep_file_meta = result.get('file_info', {})
                                    st.session_state.deep_numeric_meta = []
                                    st.session_state.deep_config_step = 5
                                    st.rerun()
                            
                            except Exception as e:
                                st.error(f"❌ API Error: {str(e)}")

            # Step 5: Text Normalization
            if st.session_state.deep_config_step == 5:
                st.subheader("Text Normalization")
                # Back button
                if st.button("Back to Stop Words", key="deep_back_to_stopwords"):
                    st.session_state.deep_chunking_result = None
                    st.session_state.deep_embedding_result = None
                    st.session_state.deep_config_step = 4
                    st.rerun()

                choice = st.radio(
                    "Choose an option:",
                    [
                        "Apply lemmatization",
                        "Apply stemming",
                        "Skip text normalization",
                    ],
                    index=2,
                    key="deep_text_norm_choice",
                )

                if st.button("Apply Changes", key="deep_apply_text_norm"):
                    with st.spinner("🔄 Applying text normalization via API..."):
                        try:
                            # Determine method based on choice
                            if choice == "Apply lemmatization":
                                method = "lemmatize"
                            elif choice == "Apply stemming":
                                method = "stem"
                            else:
                                method = "none"
                            
                            result = call_deep_config_normalize_api(method)
                            
                            if "error" in result:
                                st.error(f"❌ Text normalization failed: {result['error']}")
                            else:
                                if method == "lemmatize":
                                    st.success("✅ Applied lemmatization")
                                elif method == "stem":
                                    st.success("✅ Applied stemming")
                                else:
                                    st.info("Skipped text normalization")
                                
                                # Update session state with API results
                                st.session_state.deep_file_meta = result.get('file_info', {})
                                st.session_state.deep_numeric_meta = []
                                
                                # Add download button for preprocessed data
                                st.markdown("---")
                                st.subheader("📥 Download Preprocessed Data")
                                if st.button("📄 Download Preprocessed CSV", key="deep_download_preprocessed"):
                                    try:
                                        csv_data = download_deep_config_preprocessed()
                                        filename = "preprocessed_data.csv"
                                        st.download_button(
                                            label="⬇️ Download Preprocessed Data",
                                            data=csv_data,
                                            file_name=filename,
                                            mime="text/csv",
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.error(f"Download failed: {str(e)}")
                                
                                st.session_state.deep_config_step = 6
                                st.rerun()
                        
                        except Exception as e:
                            st.error(f"❌ API Error: {str(e)}")

            # Step 6: Final metadata and chunking
            if st.session_state.deep_config_step == 6:
                st.sidebar.checkbox("Default Preprocessing Done", value=True, disabled=True, key="deep_step6_preprocessing_done")
                st.sidebar.checkbox("Type Conversion Done", value=True, disabled=True, key="deep_step6_type_conversion_done")
                st.sidebar.checkbox("Null Handling Done", value=True, disabled=True, key="deep_step6_null_handling_done")
                
                # Back button
                if st.button("Back to Text Normalization", key="deep_back_to_text_norm"):
                    st.session_state.deep_chunking_result = None
                    st.session_state.deep_embedding_result = None
                    st.session_state.deep_config_step = 5
                    st.rerun()

                # Download preprocessed CSV
                csv_data = st.session_state.deep_df.to_csv(index=False).encode("utf-8")
                
                # Generate appropriate filename based on data source
                if uploaded_file:
                    filename = f"processed_{uploaded_file.name}"
                else:
                    # DB import case
                    table_name = st.session_state.deep_file_info.get('table', 'database_table')
                    filename = f"processed_{table_name}.csv"
                
                st.download_button(
                    label="Download preprocessed CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                )
                
                st.divider()
                
                # Metadata selection
                st.subheader("Select metadata columns to store in ChromaDB")
                
                store_metadata = st.checkbox(
                    "Store metadata in ChromaDB", 
                    value=st.session_state.deep_store_metadata_enabled, 
                    help="Enable this to store selected metadata columns in ChromaDB for filtering and retrieval",
                    key="deep_store_metadata_checkbox"
                )
                
                st.session_state.deep_store_metadata_enabled = store_metadata
                
                if store_metadata:
                    df_current = st.session_state.deep_df
                    numeric_candidates = df_current.select_dtypes(include=['number']).columns.tolist()
                    max_categorical_cardinality = 50
                    raw_categorical = df_current.select_dtypes(include=['object']).columns.tolist()
                    categorical_candidates = [c for c in raw_categorical if df_current[c].nunique(dropna=True) <= max_categorical_cardinality]

                    max_numeric_cap = min(10, len(numeric_candidates))
                    max_categorical_cap = min(5, len(categorical_candidates))

                    def _num_rank(col):
                        try:
                            var = float(pd.to_numeric(df_current[col], errors='coerce').var())
                        except Exception:
                            var = 0.0
                        miss = float(pd.to_numeric(df_current[col], errors='coerce').isna().mean())
                        return (-var, miss)
                    ranked_numeric = sorted(numeric_candidates, key=_num_rank)

                    def _cat_rank(col):
                        s = df_current[col]
                        miss = float(s.isna().mean())
                        uniq = int(s.nunique(dropna=True))
                        return (miss, uniq)
                    ranked_categorical = sorted(categorical_candidates, key=_cat_rank)

                    if numeric_candidates:
                        num_numeric_to_store = st.number_input(
                            "How many numeric columns to include (store min/mean/max per chunk)",
                            min_value=0,
                            max_value=max_numeric_cap,
                            value=max_numeric_cap,
                            key="deep_num_numeric"
                        )
                        default_numeric = ranked_numeric[: int(num_numeric_to_store)]
                        selected_numeric_cols = st.multiselect(
                            "Select numeric columns",
                            options=numeric_candidates,
                            default=(st.session_state.deep_meta_numeric_cols[: int(num_numeric_to_store)] if st.session_state.deep_meta_numeric_cols else default_numeric),
                            key="deep_selected_numeric"
                        )
                        if len(selected_numeric_cols) > int(num_numeric_to_store):
                            selected_numeric_cols = selected_numeric_cols[: int(num_numeric_to_store)]
                    else:
                        st.info("No numeric columns detected.")
                        selected_numeric_cols = []

                    if categorical_candidates:
                        num_categorical_to_store = st.number_input(
                            "How many categorical columns to include (store mode per chunk)",
                            min_value=0,
                            max_value=max_categorical_cap,
                            value=min(2, max_categorical_cap),
                            key="deep_num_categorical"
                        )
                        default_categorical = ranked_categorical[: int(num_categorical_to_store)]
                        selected_categorical_cols = st.multiselect(
                            "Select categorical columns",
                            options=categorical_candidates,
                            default=(st.session_state.deep_meta_categorical_cols[: int(num_categorical_to_store)] if st.session_state.deep_meta_categorical_cols else default_categorical),
                            key="deep_selected_categorical"
                        )
                        if len(selected_categorical_cols) > int(num_categorical_to_store):
                            selected_categorical_cols = selected_categorical_cols[: int(num_categorical_to_store)]
                        high_card = [c for c in selected_categorical_cols if df_current[c].nunique(dropna=True) > max_categorical_cardinality]
                        if high_card:
                            st.warning(f"High-cardinality categorical columns selected: {', '.join(high_card)}. This may reduce filter usefulness.")
                    else:
                        st.info("No low-cardinality categorical columns detected.")
                        selected_categorical_cols = []

                    st.session_state.deep_meta_numeric_cols = selected_numeric_cols
                    st.session_state.deep_meta_categorical_cols = selected_categorical_cols

                    if st.button("Apply Metadata Selection", key="deep_apply_metadata"):
                        st.session_state.metadata_selection_applied = True
                        st.success("Metadata selection saved. These fields will be stored with chunks in ChromaDB.")
                else:
                    st.info("Metadata storage is disabled. No metadata will be stored in ChromaDB.")
                    st.session_state.deep_meta_numeric_cols = []
                    st.session_state.deep_meta_categorical_cols = []
                
                # Add chunking section
                st.divider()
                st.subheader("CSV Chunking")
                
                if st.button("Start Chunking Process", key="deep_start_chunking"):
                    st.session_state.deep_config_step = 7
                    st.rerun()

            # Step 7: Chunking Workflow
            if st.session_state.deep_config_step == 7:
                st.divider()
                st.subheader("CSV Chunking Process")
                
                st.sidebar.checkbox("Preprocessing Complete", value=True, disabled=True, key="deep_chunking_preprocessing_complete")
                
                # Check if data is available
                if st.session_state.deep_df.empty:
                    st.error("❌ No data available for chunking. Please go back and load data first.")
                    if st.button("Back to Data Loading", key="deep_back_to_data_loading_step7"):
                        st.session_state.deep_config_step = 0
                        st.rerun()
                    st.stop()
                
                st.subheader("Select Chunking Method")

                chunking_method = st.radio(
                    "Choose a chunking method:",
                    [
                        "Fixed Size",
                        "Recursive",
                        "Semantic",
                        "Document",
                    ],
                    key="deep_chunking_method"
                )
                
                if chunking_method == "Fixed Size":
                    st.info("Splits data into fixed-size chunks of characters with overlap")
                    chunk_size = st.number_input("Chunk Size (characters)", min_value=50, max_value=20000, value=400, step=50, key="deep_fixed_chunk_size")
                    overlap = st.number_input("Overlap (characters)", min_value=0, max_value=chunk_size-1 if chunk_size>0 else 0, value=50, key="deep_fixed_overlap")

                elif chunking_method == "Recursive":
                    st.info("Splits key-value formatted lines with recursive separators and overlap")
                    chunk_size = st.number_input("Chunk Size (characters)", min_value=50, max_value=20000, value=400, step=50, key="deep_recursive_chunk_size")
                    overlap = st.number_input("Overlap (characters)", min_value=0, max_value=chunk_size-1 if chunk_size>0 else 0, value=50, key="deep_recursive_overlap")

                elif chunking_method == "Semantic":
                    st.info("Clusters rows semantically and concatenates each cluster as a chunk")
                    n_clusters = st.number_input("Number of clusters", min_value=2, max_value=max(2, len(st.session_state.deep_df)), value=10, key="deep_semantic_clusters")

                elif chunking_method == "Document":
                    st.info("Group by a key column and split by token limit (headers optional)")
                    key_column = st.selectbox("Key column", st.session_state.deep_df.columns.tolist(), key="deep_document_key_column")
                    token_limit = st.number_input("Token limit per chunk", min_value=200, max_value=10000, value=2000, step=100, key="deep_document_token_limit")
                    preserve_headers = st.checkbox("Include headers in each chunk", value=True, key="deep_document_preserve_headers")
                
                if st.button("Apply Chunking Method", key="deep_apply_chunking"):
                    with st.spinner("🔄 Applying chunking via API..."):
                        try:
                            # Prepare chunking parameters based on method
                            if chunking_method == "Fixed Size":
                                chunk_params = {
                                    "method": "fixed",
                                    "chunk_size": int(chunk_size),
                                    "overlap": int(overlap)
                                }
                            elif chunking_method == "Recursive":
                                chunk_params = {
                                    "method": "recursive",
                                    "chunk_size": int(chunk_size),
                                    "overlap": int(overlap)
                                }
                            elif chunking_method == "Semantic":
                                chunk_params = {
                                    "method": "semantic",
                                    "n_clusters": int(n_clusters)
                                }
                            elif chunking_method == "Document":
                                chunk_params = {
                                    "method": "document",
                                    "key_column": key_column,
                                    "token_limit": int(token_limit),
                                    "preserve_headers": preserve_headers
                                }
                            
                            result = call_deep_config_chunk_api(chunk_params)
                            
                            if "error" in result:
                                st.error(f"❌ Chunking failed: {result['error']}")
                            else:
                                st.success(f"✅ Successfully created {result.get('total_chunks', 'N/A')} chunks!")
                                st.info(f"📊 **Method**: {chunking_method}")
                                
                                # Update session state with API results
                                st.session_state.deep_chunking_result = {
                                    "chunks": result.get('chunks', []),
                                    "metadata": result.get('metadata', []),
                                    "method": chunking_method.lower().replace(" ", "_"),
                                    "total_chunks": result.get('total_chunks', 0)
                                }
                                
                                # Add download button for chunks
                                st.markdown("---")
                                st.subheader("📥 Download Chunks")
                                if st.button("📄 Download Chunks CSV", key="deep_download_chunks"):
                                    try:
                                        chunks_data = download_deep_config_chunks()
                                        filename = "chunks.csv"
                                        st.download_button(
                                            label="⬇️ Download Chunks",
                                            data=chunks_data,
                                            file_name=filename,
                                            mime="text/csv",
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.error(f"Download failed: {str(e)}")
                                
                                st.session_state.deep_config_step = 8
                                st.rerun()
                        
                        except Exception as e:
                            st.error(f"❌ API Error: {str(e)}")

                # Back button
                if st.button("Back to Metadata Selection", key="deep_back_to_metadata"):
                    st.session_state.deep_chunking_result = None
                    st.session_state.deep_embedding_result = None
                    st.session_state.deep_config_step = 6
                    st.rerun()

            # Step 8: Embedding Generation
            if st.session_state.deep_config_step == 8:
                st.sidebar.checkbox("Preprocessing Complete", value=True, disabled=True, key="deep_embedding_preprocessing_complete")
                st.sidebar.checkbox("Chunking Complete", value=True, disabled=True, key="deep_embedding_chunking_complete")
                
                st.subheader("Generate Embeddings")
                
                available_models = [
                    "all-MiniLM-L6-v2",
                    "paraphrase-MiniLM-L6-v2",
                    "text-embedding-ada-002"
                ]
                model_choice = st.radio(
                    "Choose an embedding model:",
                    available_models,
                    key="deep_embedding_model"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model", model_choice)
                with col2:
                    st.metric("Type", "OpenAI" if "text-embedding" in model_choice else "Local")
                
                # Configuration options
                st.subheader("Configuration")
                batch_size = st.selectbox("Batch Size", options=[32, 64, 128, 256], index=0, key="deep_embedding_batch_size", help="Larger batch sizes are faster but use more memory")
                use_parallel = st.checkbox("Use parallel encoding (local models)", value=True, key="deep_use_parallel")
                if "text-embedding" in model_choice:
                    openai_api_key = st.text_input("OpenAI API Key (optional)", type="password", key="deep_openai_api_key")
                    openai_base_url = st.text_input("OpenAI Base URL (optional)", value="", key="deep_openai_base_url")
                else:
                    openai_api_key = None
                    openai_base_url = None
                
                if st.button("Generate Embeddings", key="deep_generate_embeddings"):
                    with st.spinner("🔄 Generating embeddings via API..."):
                        try:
                            # Prepare embedding parameters
                            embed_params = {
                                "model_name": model_choice,
                                "batch_size": int(batch_size),
                                "use_parallel": bool(use_parallel)
                            }
                            
                            # Add OpenAI parameters if using OpenAI model
                            if "text-embedding" in model_choice:
                                if openai_api_key:
                                    embed_params["openai_api_key"] = openai_api_key
                                if openai_base_url:
                                    embed_params["openai_base_url"] = openai_base_url
                            
                            result = call_deep_config_embed_api(embed_params)
                            
                            if "error" in result:
                                st.error(f"❌ Embedding generation failed: {result['error']}")
                            else:
                                st.success(f"✅ Successfully generated embeddings for {result.get('total_chunks', 'N/A')} chunks!")
                                st.info(f"📊 **Model**: {model_choice}")
                                st.info(f"📊 **Vector Dimension**: {result.get('vector_dimension', 'N/A')}")
                                
                                # Update session state with API results
                                st.session_state.deep_embedding_result = {
                                    'model_used': model_choice,
                                    'total_chunks': result.get('total_chunks', 0),
                                    'vector_dimension': result.get('vector_dimension', 0),
                                    'embeddings': result.get('embeddings', []),
                                    'chunk_texts': result.get('chunk_texts', []),
                                }
                                
                                # Add download button for embeddings
                                st.markdown("---")
                                st.subheader("📥 Download Embeddings")
                                if st.button("📄 Download Embeddings JSON", key="deep_download_embeddings"):
                                    try:
                                        embeddings_data = download_deep_config_embeddings()
                                        filename = "embeddings.json"
                                        st.download_button(
                                            label="⬇️ Download Embeddings",
                                            data=embeddings_data,
                                            file_name=filename,
                                            mime="application/json",
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.error(f"Download failed: {str(e)}")
                                
                                st.session_state.deep_config_step = 9
                                st.rerun()
                        
                        except Exception as e:
                            st.error(f"❌ API Error: {str(e)}")

                # Back button
                if st.button("Back to Chunking", key="deep_back_to_chunking"):
                    st.session_state.deep_embedding_result = None
                    st.session_state.deep_config_step = 7
                    st.rerun()

            # Step 9: Storage & Retrieval
            if st.session_state.deep_config_step == 9:
                st.sidebar.checkbox("Preprocessing Complete", value=True, disabled=True, key="deep_storage_preprocessing_complete")
                st.sidebar.checkbox("Chunking Complete", value=True, disabled=True, key="deep_storage_chunking_complete")
                st.sidebar.checkbox("Embeddings Generated", value=True, disabled=True, key="deep_storage_embeddings_generated")
                
                st.subheader("Vector Storage & Retrieval")
                
                storage_choice = st.radio("Choose storage backend:", ["ChromaDB", "FAISS"], index=0, key="deep_storage_choice")
                if storage_choice == "ChromaDB":
                    default_collection = f"csv_chunks__{uploaded_file.name.replace('.csv','')}" if uploaded_file else "csv_chunks"
                    collection_name = st.text_input("Collection Name", value=default_collection, key="deep_collection_name")
                    st.session_state.collection_name = collection_name
                else:
                    st.session_state.collection_name = "csv_chunks"

                # Retrieval metric selection (shown below storage selection)
                retrieval_metric = st.selectbox(
                    "Retrieval similarity metric",
                    ["cosine", "dot", "euclidean"],
                    index=0,
                    key="deep_retrieval_metric"
                )
                # removed deep config captions per request (Config-1 only requested, keeping deep captions optional)

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Store Embeddings", key="deep_store_embeddings"):
                        with st.spinner("🔄 Storing embeddings via API..."):
                            try:
                                if st.session_state.deep_embedding_result is None:
                                    st.error("No embeddings to store. Please generate embeddings first.")
                                else:
                                    # Prepare storage parameters
                                    store_params = {
                                        "storage_type": storage_choice.lower(),
                                        "retrieval_metric": retrieval_metric
                                    }
                                    
                                    if storage_choice == "ChromaDB":
                                        store_params["collection_name"] = collection_name
                                    
                                    result = call_deep_config_store_api(store_params)
                                    
                                    if "error" in result:
                                        st.error(f"❌ Storage failed: {result['error']}")
                                    else:
                                        if storage_choice == "ChromaDB":
                                            st.success(f"✅ Stored {result.get('total_vectors', 'N/A')} vectors in ChromaDB collection '{collection_name}'.")
                                        else:
                                            st.success(f"✅ Stored {result.get('total_vectors', 'N/A')} vectors in FAISS index.")
                                        st.info(f"📊 **Storage Type**: {storage_choice}")
                                        st.info(f"📊 **Retrieval Metric**: {retrieval_metric}")
                            
                            except Exception as e:
                                st.error(f"❌ API Error: {str(e)}")
                
                with col_b:
                    query = st.text_input("Enter query to retrieve relevant chunks", value="", key="deep_retrieval_query")
                    top_k = st.slider("Top K", min_value=1, max_value=50, value=5, step=1, key="deep_top_k")
                    
                    if st.button("Search Vector DB", key="deep_search_vector_db"):
                        try:
                            if not query:
                                st.warning("Please enter a query.")
                            elif st.session_state.deep_embedding_result is None:
                                st.error("Embedding result not found. Generate embeddings first.")
                            else:
                                # Model used
                                er = st.session_state.deep_embedding_result
                                model_used = er.get('model_used', 'unknown') or 'unknown'
                                
                                # Build retriever based on storage choice
                                if storage_choice == "ChromaDB":
                                    collection_name = st.session_state.get('collection_name', 'csv_chunks')
                                    try:
                                        import chromadb
                                        client = chromadb.PersistentClient(path="chromadb_store")
                                        collection = client.get_collection(collection_name)
                                        
                                        # Generate query embedding
                                        from sentence_transformers import SentenceTransformer
                                        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                                        query_embedding = model.encode([query])
                                        
                                        # Search
                                        results = collection.query(
                                            query_embeddings=query_embedding.tolist(),
                                            n_results=int(top_k)
                                        )
                                        
                                        # Format results
                                        docs = results.get('documents', [[]])[0] if results else []
                                        metas = results.get('metadatas', [[]])[0] if results else []
                                        dists = results.get('distances', [[]])[0] if results else []
                                        
                                    except Exception as e:
                                        st.error(f"ChromaDB search failed: {e}")
                                        docs, metas, dists = [], [], []
                                else:
                                    try:
                                        import faiss
                                        import pickle
                                        
                                        # Load FAISS index and data
                                        index = faiss.read_index("faiss_store/index.faiss")
                                        with open("faiss_store/data.pkl", "rb") as f:
                                            faiss_data = pickle.load(f)
                                        
                                        # Generate query embedding
                                        from sentence_transformers import SentenceTransformer
                                        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                                        query_embedding = model.encode([query])
                                        
                                        # Search
                                        distances, indices = index.search(query_embedding, int(top_k))
                                        
                                        # Format results
                                        docs = [faiss_data['documents'][i] for i in indices[0] if i < len(faiss_data['documents'])]
                                        metas = [faiss_data['metadata'][i] for i in indices[0] if i < len(faiss_data['metadata'])]
                                        dists = distances[0].tolist()
                                        
                                    except Exception as e:
                                        st.error(f"FAISS search failed: {e}")
                                        docs, metas, dists = [], [], []
                                
                                st.success(f"Query completed successfully! (metric: {retrieval_metric})")
                                
                                st.subheader("Top Results")
                                
                                if not docs:
                                    st.info("No results found.")
                                else:
                                    st.success(f"Found {len(docs)} results!")
                                    for i, doc in enumerate(docs[: int(top_k)]):
                                        meta = metas[i] if i < len(metas) else {}
                                        score = dists[i] if i < len(dists) else None
                                        
                                        with st.expander(f"Result {i+1} — score: {score:.4f}", expanded=False):
                                            st.write(f"**Metadata:** {meta}")
                                            st.write("**Full Content:**")
                                            st.text_area(
                                                f"Chunk {i+1} Content", 
                                                value=doc, 
                                                height=300, 
                                                key=f"deep_chunk_content_{i}",
                                                help="Scroll to view the complete chunk content"
                                            )
                        except Exception as e:
                            st.error(f"Search failed: {e}")

                # Back button
                if st.button("Back to Embedding", key="deep_back_to_embedding"):
                    st.session_state.deep_config_step = 8
                    st.rerun()

                # Complete process button
                if st.button("Complete Deep Config Process", key="deep_complete_process"):
                    st.balloons()
                    st.success("🎉 Deep Config process completed successfully!")
                    st.info("You can now use the vector database for semantic search and retrieval.")


# Vector Retrieval Section with Scrollable Chunks
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
                    
                    # Display each result with scrollable chunk content
                    for i, result in enumerate(retrieval_result['results']):
                        display_scrollable_chunk(result, i)
                        
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")

# Export Section
if st.session_state.api_results:
    st.markdown("---")
    st.markdown("## 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📥 Download Chunks")
        # Config-1: export as CSV; others: TXT
        chunks_btn_label = "📄 Export Chunks as CSV" if st.session_state.current_mode == "config1" else "📄 Export Chunks as TXT"
        if st.button(chunks_btn_label, use_container_width=True):
            try:
                chunks_content = download_file("/export/chunks", "chunks.csv" if st.session_state.current_mode == "config1" else "chunks.txt")
                st.download_button(
                    label="⬇️ Download Chunks",
                    data=chunks_content,
                    file_name=("chunks.csv" if st.session_state.current_mode == "config1" else "chunks.txt"),
                    mime=("text/csv" if st.session_state.current_mode == "config1" else "text/plain"),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error exporting chunks: {str(e)}")
    
    with col2:
        st.markdown("#### 📥 Download Embeddings")
        # Config-1: export as JSON; others: TXT
        emb_btn_label = "🔢 Export Embeddings as JSON" if st.session_state.current_mode == "config1" else "🔢 Export Embeddings as TXT"
        if st.button(emb_btn_label, use_container_width=True):
            try:
                embeddings_content = download_embeddings_text()
                st.download_button(
                    label="⬇️ Download Embeddings",
                    data=embeddings_content,
                    file_name=("embeddings.json" if st.session_state.current_mode == "config1" else "embeddings.txt"),
                    mime=("application/json" if st.session_state.current_mode == "config1" else "text/plain"),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error exporting embeddings: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>📦 Chunking Optimizer v2.0 • FastAPI + Streamlit • 3GB+ File Support • Performance Optimized</p>
    <p><strong>🚀 Enhanced with Turbo Mode & Parallel Processing • 📜 Scrollable Chunk Display</strong></p>
</div>
""", unsafe_allow_html=True)