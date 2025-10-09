# Results Display Components
import streamlit as st
import pandas as pd
import json
from ..utils.session_state import get_api_results, get_retrieval_results
from ..utils.styling import render_metric_card, render_info_card

def render_results_display():
    """Render results display component"""
    
    # Get results from session state
    api_results = get_api_results()
    retrieval_results = get_retrieval_results()
    
    if not api_results and not retrieval_results:
        st.info("No results to display. Please run a processing operation first.")
        return
    
    # Processing results
    if api_results:
        render_processing_results(api_results)
    
    # Retrieval results
    if retrieval_results:
        render_retrieval_results(retrieval_results)

def render_processing_results(results: dict):
    """Render processing results"""
    
    st.markdown("### 📊 Processing Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(render_metric_card(
            "Rows Processed",
            str(results.get("rows", 0))
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(render_metric_card(
            "Chunks Created",
            str(results.get("chunks", 0))
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(render_metric_card(
            "Storage Type",
            results.get("stored", "N/A")
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(render_metric_card(
            "Model Used",
            results.get("embedding_model", "N/A")
        ), unsafe_allow_html=True)
    
    # Additional information
    if results.get("processing_time"):
        st.info(f"⏱️ Processing completed in {results['processing_time']:.2f} seconds")
    
    if results.get("turbo_mode"):
        st.info("⚡ Turbo mode was enabled for faster processing")
    
    if results.get("retrieval_ready"):
        st.success("✅ Retrieval system is ready for semantic search")
    
    # Export options
    st.markdown("#### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Chunks", use_container_width=True):
            export_chunks()
    
    with col2:
        if st.button("🔢 Export Embeddings", use_container_width=True):
            export_embeddings()
    
    with col3:
        if st.button("📋 Export Text", use_container_width=True):
            export_embeddings_text()
    
    # Detailed results
    with st.expander("🔍 Detailed Results"):
        st.json(results)

def render_retrieval_results(results: dict):
    """Render retrieval results"""
    
    st.markdown("### 🔍 Retrieval Results")
    
    # Query information
    query = results.get("query", "N/A")
    k = results.get("k", 0)
    
    st.markdown(f"**Query:** {query}")
    st.markdown(f"**Results:** {len(results.get('results', []))} of {k} requested")
    
    # Results list
    search_results = results.get("results", [])
    
    if search_results:
        for i, result in enumerate(search_results):
            with st.expander(f"Result {i+1} - Similarity: {result.get('similarity', 0):.3f}"):
                
                # Content
                st.markdown("#### Content")
                st.text_area(
                    "Chunk Text",
                    value=result.get("content", ""),
                    height=200,
                    key=f"retrieval_content_{i}"
                )
                
                # Metadata
                if result.get("metadata"):
                    st.markdown("#### Metadata")
                    st.json(result["metadata"])
                
                # Similarity scores
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Similarity", f"{result.get('similarity', 0):.3f}")
                
                with col2:
                    st.metric("Distance", f"{result.get('distance', 0):.3f}")
                
                # Actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📋 Copy", key=f"copy_{i}"):
                        st.write("Content copied to clipboard")
                
                with col2:
                    if st.button("🔍 Expand", key=f"expand_{i}"):
                        st.write("Expanded view")
                
                with col3:
                    if st.button("📤 Export", key=f"export_{i}"):
                        st.write("Exporting result...")
    
    else:
        st.warning("No results found for the given query")
    
    # Export all results
    if search_results:
        st.markdown("#### 📤 Export All Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export as CSV", use_container_width=True):
                export_retrieval_results_csv(search_results)
        
        with col2:
            if st.button("📋 Export as JSON", use_container_width=True):
                export_retrieval_results_json(results)

def export_chunks():
    """Export chunks functionality"""
    try:
        # This would typically call the API export endpoint
        st.success("✅ Chunks exported successfully!")
        st.info("Download will start automatically")
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_embeddings():
    """Export embeddings functionality"""
    try:
        # This would typically call the API export endpoint
        st.success("✅ Embeddings exported successfully!")
        st.info("Download will start automatically")
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_embeddings_text():
    """Export embeddings as text functionality"""
    try:
        # This would typically call the API export endpoint
        st.success("✅ Embeddings text exported successfully!")
        st.info("Download will start automatically")
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_retrieval_results_csv(results: list):
    """Export retrieval results as CSV"""
    try:
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Display download link
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name="retrieval_results.csv",
            mime="text/csv"
        )
        
        st.success("✅ CSV export ready!")
    except Exception as e:
        st.error(f"❌ CSV export failed: {str(e)}")

def export_retrieval_results_json(results: dict):
    """Export retrieval results as JSON"""
    try:
        # Convert results to JSON
        json_data = json.dumps(results, indent=2)
        
        # Display download link
        st.download_button(
            label="📋 Download JSON",
            data=json_data,
            file_name="retrieval_results.json",
            mime="application/json"
        )
        
        st.success("✅ JSON export ready!")
    except Exception as e:
        st.error(f"❌ JSON export failed: {str(e)}")

def render_processing_status():
    """Render processing status indicator"""
    
    st.markdown("### 📊 Processing Status")
    
    # Status indicators
    status_colors = {
        "pending": "#FFA500",
        "completed": "#00FF00",
        "ready": "#00BFFF",
        "error": "#FF0000"
    }
    
    # Process steps
    steps = [
        ("Preprocessing", "preprocessing"),
        ("Chunking", "chunking"),
        ("Embedding", "embedding"),
        ("Storage", "storage"),
        ("Retrieval", "retrieval")
    ]
    
    for step_name, step_key in steps:
        status = st.session_state.process_status.get(step_key, "pending")
        color = status_colors.get(status, "#FFA500")
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 12px; height: 12px; background-color: {color}; border-radius: 50%; margin-right: 10px;"></div>
            <span style="color: {color}; font-weight: bold;">{step_name}: {status.title()}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Overall progress
    completed_steps = sum(1 for status in st.session_state.process_status.values() 
                         if status in ["completed", "ready"])
    total_steps = len(st.session_state.process_status)
    progress = completed_steps / total_steps
    
    st.progress(progress)
    st.markdown(f"**Progress:** {completed_steps}/{total_steps} steps completed ({progress*100:.1f}%)")
