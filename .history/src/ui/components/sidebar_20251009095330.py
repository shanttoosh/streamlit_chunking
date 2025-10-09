# Sidebar Components
import streamlit as st
import requests
from ..utils.api_client import APIClient
from ..utils.session_state import get_process_status, update_process_status
from ..utils.styling import render_status_indicator, render_metric_card

def render_sidebar():
    """Render the main sidebar with process tracking and system info"""
    
    with st.sidebar:
        # Process Tracker Header
        st.markdown("""
        <div class="process-tracker">
            <h2>Process Tracker</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # API connection test
        api_client = APIClient()
        try:
            health_response = api_client.health_check()
            if health_response.get("success"):
                st.success("✅ API Connected")
                
                # Show capabilities
                capabilities_response = api_client.get_capabilities()
                if capabilities_response.get("success"):
                    capabilities = capabilities_response["data"]
                    if capabilities.get('large_file_support'):
                        st.info("🚀 3GB+ File Support")
                    if capabilities.get('performance_features', {}).get('turbo_mode'):
                        st.info("⚡ Turbo Mode Available")
            else:
                st.error("❌ API Not Connected")
        except:
            st.error("❌ API Not Connected")
        
        st.markdown("---")
        
        # OpenAI Configuration
        with st.expander("🔑 OpenAI Configuration"):
            st.session_state.use_openai = st.checkbox("Use OpenAI API", value=st.session_state.use_openai)
            if st.session_state.use_openai:
                st.session_state.openai_api_key = st.text_input("API Key", type="password", value=st.session_state.openai_api_key)
                st.session_state.openai_base_url = st.text_input("Base URL (optional)", value=st.session_state.openai_base_url)
        
        st.markdown("---")
        
        # Process Status Tracking
        st.markdown("### 📊 Process Status")
        
        # Preprocessing status
        preprocessing_status = get_process_status("preprocessing")
        st.markdown(render_status_indicator(preprocessing_status, "Preprocessing"), unsafe_allow_html=True)
        
        # Chunking status
        chunking_status = get_process_status("chunking")
        st.markdown(render_status_indicator(chunking_status, "Chunking"), unsafe_allow_html=True)
        
        # Embedding status
        embedding_status = get_process_status("embedding")
        st.markdown(render_status_indicator(embedding_status, "Embedding"), unsafe_allow_html=True)
        
        # Storage status
        storage_status = get_process_status("storage")
        st.markdown(render_status_indicator(storage_status, "Storage"), unsafe_allow_html=True)
        
        # Retrieval status
        retrieval_status = get_process_status("retrieval")
        st.markdown(render_status_indicator(retrieval_status, "Retrieval"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # System Information
        st.markdown("### 💻 System Info")
        
        try:
            system_response = api_client.get_system_info()
            if system_response.get("success"):
                system_info = system_response["data"]
                
                # Memory usage
                st.markdown(render_metric_card(
                    "Memory Usage", 
                    system_info.get("memory_usage", "N/A"),
                    system_info.get("available_memory", "")
                ), unsafe_allow_html=True)
                
                # Large file support
                if system_info.get("large_file_support"):
                    st.info("🚀 Large File Support Enabled")
                
                # Performance features
                st.markdown(render_metric_card(
                    "Batch Size",
                    str(system_info.get("embedding_batch_size", 256)),
                    "Embedding Processing"
                ), unsafe_allow_html=True)
                
                st.markdown(render_metric_card(
                    "Workers",
                    str(system_info.get("parallel_workers", 6)),
                    "Parallel Processing"
                ), unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"Could not load system info: {e}")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        if st.button("📊 View Results", use_container_width=True):
            if st.session_state.api_results:
                st.success("Results available in main area")
            else:
                st.warning("No results available")
        
        if st.button("🔍 Test Retrieval", use_container_width=True):
            if get_process_status("retrieval") == "ready":
                st.success("Retrieval system ready")
            else:
                st.warning("Retrieval system not ready")
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div class="footer">
            <strong>Chunk Optimizer v2.0</strong><br>
            Advanced Text Processing
        </div>
        """, unsafe_allow_html=True)
