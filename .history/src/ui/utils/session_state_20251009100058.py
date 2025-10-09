# Session State Management
import streamlit as st
from typing import Dict, Any

def initialize_session_state():
    """Initialize all session state variables"""
    
    # API results
    if "api_results" not in st.session_state:
        st.session_state.api_results = None
    
    # Current mode
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = None
    
    # File upload
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    
    # Retrieval results
    if "retrieval_results" not in st.session_state:
        st.session_state.retrieval_results = None
    
    # Process status tracking
    if "process_status" not in st.session_state:
        st.session_state.process_status = {
            "preprocessing": "pending",
            "chunking": "pending", 
            "embedding": "pending",
            "storage": "pending",
            "retrieval": "pending"
        }
    
    # Process timings
    if "process_timings" not in st.session_state:
        st.session_state.process_timings = {}
    
    # File information
    if "file_info" not in st.session_state:
        st.session_state.file_info = {}
    
    # Current dataframe
    if "current_df" not in st.session_state:
        st.session_state.current_df = None
    
    # Column types
    if "column_types" not in st.session_state:
        st.session_state.column_types = {}
    
    # Preview dataframe
    if "preview_df" not in st.session_state:
        st.session_state.preview_df = None
    
    # Text processing options
    if "text_processing_option" not in st.session_state:
        st.session_state.text_processing_option = "none"
    
    # Preview updated flag
    if "preview_updated" not in st.session_state:
        st.session_state.preview_updated = False
    
    # OpenAI configuration
    if "use_openai" not in st.session_state:
        st.session_state.use_openai = False
    
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    
    if "openai_base_url" not in st.session_state:
        st.session_state.openai_base_url = ""
    
    # Large file processing
    if "process_large_files" not in st.session_state:
        st.session_state.process_large_files = True
    
    # Temporary file path
    if "temp_file_path" not in st.session_state:
        st.session_state.temp_file_path = None
    
    # Turbo mode
    if "use_turbo" not in st.session_state:
        st.session_state.use_turbo = True
    
    # Batch size
    if "batch_size" not in st.session_state:
        st.session_state.batch_size = 256
    
    # Deep config specific
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
        st.session_state.deep_df = None
    
    if "deep_file_meta" not in st.session_state:
        st.session_state.deep_file_meta = {}
    
    if "deep_numeric_meta" not in st.session_state:
        st.session_state.deep_numeric_meta = []
    
    if "deep_chunks" not in st.session_state:
        st.session_state.deep_chunks = []
    
    if "deep_embeddings" not in st.session_state:
        st.session_state.deep_embeddings = None
    
    if "deep_model" not in st.session_state:
        st.session_state.deep_model = None
    
    if "deep_store_info" not in st.session_state:
        st.session_state.deep_store_info = None

def reset_session_state():
    """Reset all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize
    initialize_session_state()

def update_process_status(step: str, status: str):
    """Update process status for a specific step"""
    st.session_state.process_status[step] = status

def get_process_status(step: str) -> str:
    """Get process status for a specific step"""
    return st.session_state.process_status.get(step, "pending")

def is_processing_complete() -> bool:
    """Check if all processing steps are complete"""
    required_steps = ["preprocessing", "chunking", "embedding", "storage"]
    return all(
        st.session_state.process_status.get(step) in ["completed", "ready"] 
        for step in required_steps
    )

def set_file_info(file_info: Dict[str, Any]):
    """Set file information in session state"""
    st.session_state.file_info = file_info

def get_file_info() -> Dict[str, Any]:
    """Get file information from session state"""
    return st.session_state.file_info

def set_api_results(results: Dict[str, Any]):
    """Set API results in session state"""
    st.session_state.api_results = results

def get_api_results() -> Dict[str, Any]:
    """Get API results from session state"""
    return st.session_state.api_results

def set_retrieval_results(results: Dict[str, Any]):
    """Set retrieval results in session state"""
    st.session_state.retrieval_results = results

def get_retrieval_results() -> Dict[str, Any]:
    """Get retrieval results from session state"""
    return st.session_state.retrieval_results
