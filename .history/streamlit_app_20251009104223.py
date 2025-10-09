# Standalone Streamlit Application
import streamlit as st
import tempfile
import os
import sys
import requests
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import components
from src.ui.components.sidebar import render_sidebar
from src.ui.components.file_upload import render_file_upload
from src.ui.components.database_config import render_database_config
from src.ui.components.processing_config import render_processing_config
from src.ui.components.results_display import render_results_display
from src.ui.components.deep_config import render_deep_config
from src.ui.utils.api_client import APIClient
from src.ui.utils.session_state import (
    initialize_session_state, set_current_mode, get_current_mode,
    set_uploaded_file, get_uploaded_file, get_db_config, get_openai_config,
    get_processing_options, set_api_results, get_api_results,
    update_process_status, reset_process_status
)
from src.ui.utils.styling import apply_custom_styling, render_logo

def main():
    """Main Streamlit application"""
    # Apply custom styling
    apply_custom_styling()
    
    # Initialize session state
    initialize_session_state()
    
    # Set page config
    st.set_page_config(
        page_title="Chunk Optimizer", 
        layout="wide", 
        page_icon="🧠"
    )
    
    # Render logo in sidebar
    render_logo()
    
    # Main title
    st.title("🧠 Chunk Optimizer")
    st.markdown("Advanced text chunking and embedding system with multiple processing modes")
    
    # Render sidebar
    render_sidebar()
    
    # Mode selection
    st.markdown("### Select Processing Mode")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚡ Fast Mode", use_container_width=True, type="primary"):
            set_current_mode("fast")
            reset_process_status()
    
    with col2:
        if st.button("⚙️ Config-1 Mode", use_container_width=True, type="primary"):
            set_current_mode("config1")
            reset_process_status()
    
    with col3:
        if st.button("🔬 Deep Config Mode", use_container_width=True, type="primary"):
            set_current_mode("deep")
            reset_process_status()
    
    # Mode-specific processing
    current_mode = get_current_mode()
    if current_mode:
        if current_mode == "fast":
            render_fast_mode()
        elif current_mode == "config1":
            render_config1_mode()
        elif current_mode == "deep":
            render_deep_mode()
    
    # Footer
    st.markdown("---")
    st.markdown("**Chunk Optimizer v2.0** - Advanced text processing and embedding system")

def render_fast_mode():
    """Render Fast Mode interface"""
    st.markdown("### ⚡ Fast Mode Configuration")
    
    # Input source selection
    input_source = st.radio("Select Input Source:", ["📁 Upload CSV File", "🗄️ Database Import"], key="fast_input_source")
    
    if input_source == "📁 Upload CSV File":
        render_file_upload("fast")
    else:
        render_database_config("fast")
    
    # Processing options
    st.markdown("#### Processing Options")
    col1, col2 = st.columns(2)
    
    with col1:
        use_openai = st.checkbox("Use OpenAI API", key="fast_openai")
        if use_openai:
            openai_api_key = st.text_input("OpenAI API Key", type="password", key="fast_api_key")
            openai_base_url = st.text_input("OpenAI Base URL (optional)", key="fast_base_url")
        else:
            openai_api_key = ""
            openai_base_url = ""
    
    with col2:
        use_turbo = st.checkbox("Enable Turbo Mode", value=True, key="fast_turbo")
        batch_size = st.slider("Batch Size", 64, 512, 256, key="fast_batch_size")
    
    # Process button
    if st.button("🚀 Process with Fast Mode", use_container_width=True, type="primary"):
        process_fast_mode(use_openai, openai_api_key, openai_base_url, use_turbo, batch_size)

def render_config1_mode():
    """Render Config-1 Mode interface"""
    st.markdown("### ⚙️ Config-1 Mode Configuration")
    
    # Input source selection
    input_source = st.radio("Select Input Source:", ["📁 Upload CSV File", "🗄️ Database Import"], key="config1_input_source")
    
    if input_source == "📁 Upload CSV File":
        render_file_upload("config1")
    else:
        render_database_config("config1")
    
    # Chunking configuration
    st.markdown("#### Chunking Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_method = st.selectbox("Chunking Method", ["fixed", "recursive", "semantic", "document"], key="config1_chunk_method")
        chunk_size = st.slider("Chunk Size", 100, 1000, 400, key="config1_chunk_size")
        overlap = st.slider("Overlap", 0, 200, 50, key="config1_overlap")
    
    with col2:
        model_choice = st.selectbox("Embedding Model", ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L6-v2", "text-embedding-ada-002"], key="config1_model")
        storage_choice = st.selectbox("Storage", ["faiss", "chroma"], key="config1_storage")
        retrieval_metric = st.selectbox("Retrieval Metric", ["cosine", "dot", "euclidean"], key="config1_metric")
    
    # Advanced options
    with st.expander("Advanced Options"):
        use_openai = st.checkbox("Use OpenAI API", key="config1_openai")
        if use_openai:
            openai_api_key = st.text_input("OpenAI API Key", type="password", key="config1_api_key")
            openai_base_url = st.text_input("OpenAI Base URL (optional)", key="config1_base_url")
        else:
            openai_api_key = ""
            openai_base_url = ""
        
        use_turbo = st.checkbox("Enable Turbo Mode", key="config1_turbo")
        batch_size = st.slider("Batch Size", 64, 512, 256, key="config1_batch_size")
    
    # Process button
    if st.button("⚙️ Process with Config-1 Mode", use_container_width=True, type="primary"):
        config = {
            "chunk_method": chunk_method,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "model_choice": model_choice,
            "storage_choice": storage_choice,
            "retrieval_metric": retrieval_metric
        }
        process_config1_mode(config, use_openai, openai_api_key, openai_base_url, use_turbo, batch_size)

def render_deep_mode():
    """Render Deep Config Mode interface"""
    st.markdown("### 🔬 Deep Config Mode - Comprehensive Workflow")
    
    # Use the deep config component
    render_deep_config()

def process_fast_mode(use_openai=False, openai_api_key="", openai_base_url="", use_turbo=False, batch_size=256):
    """Process with Fast Mode"""
    try:
        # Initialize API client
        api_client = APIClient()
        
        # Get uploaded file
        uploaded_file = get_uploaded_file()
        db_config = get_db_config()
        
        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Call API
                with st.spinner("Processing with Fast Mode..."):
                    result = api_client.call_fast_api(
                        tmp_file_path, uploaded_file.name, "sqlite",
                        use_openai=use_openai, openai_api_key=openai_api_key,
                        openai_base_url=openai_base_url, use_turbo=use_turbo, batch_size=batch_size
                    )
                
                if result.get("success") or "result" in result:
                    set_api_results(result)
                    update_process_status("preprocessing", "completed")
                    update_process_status("chunking", "completed")
                    update_process_status("embedding", "completed")
                    update_process_status("storage", "completed")
                    update_process_status("retrieval", "ready")
                    st.success("✅ Fast Mode processing completed successfully!")
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        elif db_config.get('use_db'):
            # Call API for database import
            with st.spinner("Processing with Fast Mode..."):
                result = api_client.call_fast_api(
                    "", "", db_config.get('db_type', 'mysql'), db_config,
                    use_openai=use_openai, openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url, use_turbo=use_turbo, batch_size=batch_size
                )
            
            if result.get("success") or "result" in result:
                set_api_results(result)
                update_process_status("preprocessing", "completed")
                update_process_status("chunking", "completed")
                update_process_status("embedding", "completed")
                update_process_status("storage", "completed")
                update_process_status("retrieval", "ready")
                st.success("✅ Fast Mode processing completed successfully!")
            else:
                st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        else:
            st.error("❌ Please upload a file or configure database connection")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def process_config1_mode(config, use_openai=False, openai_api_key="", openai_base_url="", use_turbo=False, batch_size=256):
    """Process with Config-1 Mode"""
    try:
        # Initialize API client
        api_client = APIClient()
        
        # Get uploaded file
        uploaded_file = get_uploaded_file()
        db_config = get_db_config()
        
        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Call API
                with st.spinner("Processing with Config-1 Mode..."):
                    result = api_client.call_config1_api(
                        tmp_file_path, uploaded_file.name, config,
                        use_openai=use_openai, openai_api_key=openai_api_key,
                        openai_base_url=openai_base_url, use_turbo=use_turbo, batch_size=batch_size
                    )
                
                if result.get("success") or "result" in result:
                    set_api_results(result)
                    update_process_status("preprocessing", "completed")
                    update_process_status("chunking", "completed")
                    update_process_status("embedding", "completed")
                    update_process_status("storage", "completed")
                    update_process_status("retrieval", "ready")
                    st.success("✅ Config-1 Mode processing completed successfully!")
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        elif db_config.get('use_db'):
            # Call API for database import
            with st.spinner("Processing with Config-1 Mode..."):
                result = api_client.call_config1_api(
                    "", "", config, db_config,
                    use_openai=use_openai, openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url, use_turbo=use_turbo, batch_size=batch_size
                )
            
            if result.get("success") or "result" in result:
                set_api_results(result)
                update_process_status("preprocessing", "completed")
                update_process_status("chunking", "completed")
                update_process_status("embedding", "completed")
                update_process_status("storage", "completed")
                update_process_status("retrieval", "ready")
                st.success("✅ Config-1 Mode processing completed successfully!")
            else:
                st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        else:
            st.error("❌ Please upload a file or configure database connection")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
