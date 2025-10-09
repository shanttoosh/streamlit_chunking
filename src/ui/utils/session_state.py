# Session State Management for Streamlit UI
import streamlit as st
from typing import Dict, Any, Optional

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    # Processing status tracking
    if 'process_status' not in st.session_state:
        st.session_state.process_status = {
            "preprocessing": "pending",
            "chunking": "pending", 
            "embedding": "pending",
            "storage": "pending",
            "retrieval": "pending"
        }
    
    # Current mode
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = None
    
    # File upload state
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    
    # Database configuration
    if 'db_config' not in st.session_state:
        st.session_state.db_config = {
            'use_db': False,
            'db_type': 'mysql',
            'host': '',
            'port': 3306,
            'username': '',
            'password': '',
            'database': '',
            'table_name': ''
        }
    
    # OpenAI configuration
    if 'use_openai' not in st.session_state:
        st.session_state.use_openai = False
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''
    if 'openai_base_url' not in st.session_state:
        st.session_state.openai_base_url = ''
    
    # Processing options
    if 'use_turbo' not in st.session_state:
        st.session_state.use_turbo = False
    if 'batch_size' not in st.session_state:
        st.session_state.batch_size = 256
    
    # API results
    if 'api_results' not in st.session_state:
        st.session_state.api_results = None
    
    # Deep config state
    if 'deep_config_step' not in st.session_state:
        st.session_state.deep_config_step = 1
    if 'deep_config_data' not in st.session_state:
        st.session_state.deep_config_data = {}
    if 'deep_config_preprocessed' not in st.session_state:
        st.session_state.deep_config_preprocessed = None
    if 'deep_config_chunks' not in st.session_state:
        st.session_state.deep_config_chunks = None
    if 'deep_config_embeddings' not in st.session_state:
        st.session_state.deep_config_embeddings = None
    
    # Deep config metadata
    if 'deep_meta_numeric_cols' not in st.session_state:
        st.session_state.deep_meta_numeric_cols = []
    if 'deep_meta_categorical_cols' not in st.session_state:
        st.session_state.deep_meta_categorical_cols = []
    
    # Retrieval state
    if 'retrieval_query' not in st.session_state:
        st.session_state.retrieval_query = ''
    if 'retrieval_results' not in st.session_state:
        st.session_state.retrieval_results = None
    if 'retrieval_k' not in st.session_state:
        st.session_state.retrieval_k = 5
    
    # System info
    if 'system_info' not in st.session_state:
        st.session_state.system_info = None
    if 'file_info' not in st.session_state:
        st.session_state.file_info = None
    if 'capabilities' not in st.session_state:
        st.session_state.capabilities = None

def update_process_status(step: str, status: str):
    """Update processing status"""
    st.session_state.process_status[step] = status

def get_process_status(step: str) -> str:
    """Get processing status for a step"""
    return st.session_state.process_status.get(step, "pending")

def reset_process_status():
    """Reset all processing status to pending"""
    st.session_state.process_status = {
        "preprocessing": "pending",
        "chunking": "pending", 
        "embedding": "pending",
        "storage": "pending",
        "retrieval": "pending"
    }

def set_current_mode(mode: str):
    """Set current processing mode"""
    st.session_state.current_mode = mode
    reset_process_status()

def get_current_mode() -> Optional[str]:
    """Get current processing mode"""
    return st.session_state.current_mode

def set_uploaded_file(file):
    """Set uploaded file"""
    st.session_state.uploaded_file = file

def get_uploaded_file():
    """Get uploaded file"""
    return st.session_state.uploaded_file

def set_db_config(config: Dict[str, Any]):
    """Set database configuration"""
    st.session_state.db_config.update(config)

def get_db_config() -> Dict[str, Any]:
    """Get database configuration"""
    return st.session_state.db_config

def set_openai_config(use_openai: bool, api_key: str = '', base_url: str = ''):
    """Set OpenAI configuration"""
    st.session_state.use_openai = use_openai
    st.session_state.openai_api_key = api_key
    st.session_state.openai_base_url = base_url

def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration"""
    return {
        'use_openai': st.session_state.use_openai,
        'api_key': st.session_state.openai_api_key,
        'base_url': st.session_state.openai_base_url
    }

def set_processing_options(use_turbo: bool, batch_size: int):
    """Set processing options"""
    st.session_state.use_turbo = use_turbo
    st.session_state.batch_size = batch_size

def get_processing_options() -> Dict[str, Any]:
    """Get processing options"""
    return {
        'use_turbo': st.session_state.use_turbo,
        'batch_size': st.session_state.batch_size
    }

def set_api_results(results: Dict[str, Any]):
    """Set API results"""
    st.session_state.api_results = results

def get_api_results() -> Optional[Dict[str, Any]]:
    """Get API results"""
    return st.session_state.api_results

def set_deep_config_step(step: int):
    """Set deep config step"""
    st.session_state.deep_config_step = step

def get_deep_config_step() -> int:
    """Get deep config step"""
    return st.session_state.deep_config_step

def set_deep_config_data(key: str, value: Any):
    """Set deep config data"""
    st.session_state.deep_config_data[key] = value

def get_deep_config_data(key: str, default: Any = None) -> Any:
    """Get deep config data"""
    return st.session_state.deep_config_data.get(key, default)

def set_deep_config_preprocessed(data):
    """Set deep config preprocessed data"""
    st.session_state.deep_config_preprocessed = data

def get_deep_config_preprocessed():
    """Get deep config preprocessed data"""
    return st.session_state.deep_config_preprocessed

def set_deep_config_chunks(chunks):
    """Set deep config chunks"""
    st.session_state.deep_config_chunks = chunks

def get_deep_config_chunks():
    """Get deep config chunks"""
    return st.session_state.deep_config_chunks

def set_deep_config_embeddings(embeddings):
    """Set deep config embeddings"""
    st.session_state.deep_config_embeddings = embeddings

def get_deep_config_embeddings():
    """Get deep config embeddings"""
    return st.session_state.deep_config_embeddings

def set_deep_metadata(numeric_cols: list, categorical_cols: list):
    """Set deep config metadata columns"""
    st.session_state.deep_meta_numeric_cols = numeric_cols
    st.session_state.deep_meta_categorical_cols = categorical_cols

def get_deep_metadata() -> Dict[str, list]:
    """Get deep config metadata columns"""
    return {
        'numeric_cols': st.session_state.deep_meta_numeric_cols,
        'categorical_cols': st.session_state.deep_meta_categorical_cols
    }

def set_retrieval_query(query: str):
    """Set retrieval query"""
    st.session_state.retrieval_query = query

def get_retrieval_query() -> str:
    """Get retrieval query"""
    return st.session_state.retrieval_query

def set_retrieval_results(results: Dict[str, Any]):
    """Set retrieval results"""
    st.session_state.retrieval_results = results

def get_retrieval_results() -> Optional[Dict[str, Any]]:
    """Get retrieval results"""
    return st.session_state.retrieval_results

def set_retrieval_k(k: int):
    """Set retrieval k value"""
    st.session_state.retrieval_k = k

def get_retrieval_k() -> int:
    """Get retrieval k value"""
    return st.session_state.retrieval_k

def set_system_info(info: Dict[str, Any]):
    """Set system information"""
    st.session_state.system_info = info

def get_system_info() -> Optional[Dict[str, Any]]:
    """Get system information"""
    return st.session_state.system_info

def set_file_info(info: Dict[str, Any]):
    """Set file information"""
    st.session_state.file_info = info

def get_file_info() -> Optional[Dict[str, Any]]:
    """Get file information"""
    return st.session_state.file_info

def set_capabilities(caps: Dict[str, Any]):
    """Set system capabilities"""
    st.session_state.capabilities = caps

def get_capabilities() -> Optional[Dict[str, Any]]:
    """Get system capabilities"""
    return st.session_state.capabilities

def clear_session_state():
    """Clear all session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()