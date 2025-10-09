# Main Streamlit Application
import streamlit as st
from .components.sidebar import render_sidebar
from .components.file_upload import render_file_upload
from .components.database_config import render_database_config
from .components.processing_config import render_processing_config
from .components.results_display import render_results_display
from .components.deep_config import render_deep_config
from .utils.api_client import APIClient
from .utils.session_state import initialize_session_state
from .utils.styling import apply_custom_styling

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
            st.session_state.current_mode = "fast"
            st.session_state.process_status = {k: "pending" for k in st.session_state.process_status}
    
    with col2:
        if st.button("⚙️ Config-1 Mode", use_container_width=True, type="primary"):
            st.session_state.current_mode = "config1"
            st.session_state.process_status = {k: "pending" for k in st.session_state.process_status}
    
    with col3:
        if st.button("🔬 Deep Config Mode", use_container_width=True, type="primary"):
            st.session_state.current_mode = "deep"
            st.session_state.process_status = {k: "pending" for k in st.session_state.process_status}
    
    # Mode-specific processing
    if st.session_state.current_mode:
        if st.session_state.current_mode == "fast":
            render_fast_mode()
        elif st.session_state.current_mode == "config1":
            render_config1_mode()
        elif st.session_state.current_mode == "deep":
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
        st.session_state.use_openai = st.checkbox("Use OpenAI API", key="fast_openai")
        if st.session_state.use_openai:
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", key="fast_api_key")
            st.session_state.openai_base_url = st.text_input("OpenAI Base URL (optional)", key="fast_base_url")
    
    with col2:
        st.session_state.use_turbo = st.checkbox("Enable Turbo Mode", value=True, key="fast_turbo")
        st.session_state.batch_size = st.slider("Batch Size", 64, 512, 256, key="fast_batch_size")
    
    # Process button
    if st.button("🚀 Process with Fast Mode", use_container_width=True, type="primary"):
        process_fast_mode()

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
        st.session_state.use_openai = st.checkbox("Use OpenAI API", key="config1_openai")
        if st.session_state.use_openai:
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", key="config1_api_key")
            st.session_state.openai_base_url = st.text_input("OpenAI Base URL (optional)", key="config1_base_url")
        
        st.session_state.use_turbo = st.checkbox("Enable Turbo Mode", key="config1_turbo")
        st.session_state.batch_size = st.slider("Batch Size", 64, 512, 256, key="config1_batch_size")
    
    # Process button
    if st.button("⚙️ Process with Config-1 Mode", use_container_width=True, type="primary"):
        process_config1_mode()

def render_deep_mode():
    """Render Deep Config Mode interface"""
    st.markdown("### 🔬 Deep Config Mode - Comprehensive Workflow")
    
    # Use the deep config component
    render_deep_config()

def process_fast_mode():
    """Process with Fast Mode"""
    try:
        # Initialize API client
        api_client = APIClient()
        
        # Prepare request data
        data = {
            "use_openai": st.session_state.use_openai,
            "openai_api_key": st.session_state.openai_api_key,
            "openai_base_url": st.session_state.openai_base_url,
            "use_turbo": st.session_state.use_turbo,
            "batch_size": st.session_state.batch_size
        }
        
        # Add file if uploaded
        if st.session_state.uploaded_file:
            data["file"] = st.session_state.uploaded_file
        
        # Call API
        with st.spinner("Processing with Fast Mode..."):
            result = api_client.process_fast(data)
        
        if result.get("success"):
            st.session_state.api_results = result
            st.session_state.process_status["preprocessing"] = "completed"
            st.session_state.process_status["chunking"] = "completed"
            st.session_state.process_status["embedding"] = "completed"
            st.session_state.process_status["storage"] = "completed"
            st.session_state.process_status["retrieval"] = "ready"
            st.success("✅ Fast Mode processing completed successfully!")
        else:
            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def process_config1_mode():
    """Process with Config-1 Mode"""
    try:
        # Initialize API client
        api_client = APIClient()
        
        # Prepare request data
        data = {
            "chunk_method": st.session_state.get("config1_chunk_method", "recursive"),
            "chunk_size": st.session_state.get("config1_chunk_size", 400),
            "overlap": st.session_state.get("config1_overlap", 50),
            "model_choice": st.session_state.get("config1_model", "paraphrase-MiniLM-L6-v2"),
            "storage_choice": st.session_state.get("config1_storage", "faiss"),
            "retrieval_metric": st.session_state.get("config1_metric", "cosine"),
            "use_openai": st.session_state.use_openai,
            "openai_api_key": st.session_state.openai_api_key,
            "openai_base_url": st.session_state.openai_base_url,
            "use_turbo": st.session_state.use_turbo,
            "batch_size": st.session_state.batch_size
        }
        
        # Add file if uploaded
        if st.session_state.uploaded_file:
            data["file"] = st.session_state.uploaded_file
        
        # Call API
        with st.spinner("Processing with Config-1 Mode..."):
            result = api_client.process_config1(data)
        
        if result.get("success"):
            st.session_state.api_results = result
            st.session_state.process_status["preprocessing"] = "completed"
            st.session_state.process_status["chunking"] = "completed"
            st.session_state.process_status["embedding"] = "completed"
            st.session_state.process_status["storage"] = "completed"
            st.session_state.process_status["retrieval"] = "ready"
            st.success("✅ Config-1 Mode processing completed successfully!")
        else:
            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
