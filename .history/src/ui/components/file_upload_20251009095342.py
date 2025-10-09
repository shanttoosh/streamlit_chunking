# File Upload Components
import streamlit as st
import pandas as pd
from ..utils.session_state import set_file_info, get_file_info
from ..utils.styling import render_info_card

def render_file_upload(mode: str = "fast"):
    """Render file upload component"""
    
    st.markdown("#### 📁 File Upload")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        key=f"{mode}_file_uploader",
        help="Upload a CSV file for processing"
    )
    
    if uploaded_file is not None:
        # Store uploaded file in session state
        st.session_state.uploaded_file = uploaded_file
        
        # Display file information
        file_info = {
            "filename": uploaded_file.name,
            "size": uploaded_file.size,
            "type": uploaded_file.type
        }
        
        set_file_info(file_info)
        
        # Show file details
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(render_info_card(
                "Filename",
                uploaded_file.name
            ), unsafe_allow_html=True)
        
        with col2:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(render_info_card(
                "File Size",
                f"{file_size_mb:.2f} MB"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(render_info_card(
                "File Type",
                uploaded_file.type or "CSV"
            ), unsafe_allow_html=True)
        
        # Preview data
        try:
            # Read CSV preview
            df_preview = pd.read_csv(uploaded_file, nrows=5)
            
            st.markdown("#### 📋 Data Preview")
            st.dataframe(df_preview, use_container_width=True)
            
            # Show data info
            st.markdown("#### 📊 Data Information")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", len(df_preview))
            
            with col2:
                st.metric("Columns", len(df_preview.columns))
            
            with col3:
                # Count non-null values
                non_null_count = df_preview.count().sum()
                st.metric("Non-Null Values", non_null_count)
            
            with col4:
                # Count null values
                null_count = df_preview.isnull().sum().sum()
                st.metric("Null Values", null_count)
            
            # Column information
            st.markdown("#### 📝 Column Information")
            
            column_info = []
            for col in df_preview.columns:
                col_info = {
                    "Column": col,
                    "Type": str(df_preview[col].dtype),
                    "Non-Null": df_preview[col].count(),
                    "Null": df_preview[col].isnull().sum(),
                    "Unique": df_preview[col].nunique()
                }
                column_info.append(col_info)
            
            column_df = pd.DataFrame(column_info)
            st.dataframe(column_df, use_container_width=True)
            
            # Store preview in session state
            st.session_state.preview_df = df_preview
            
            # Large file warning
            if file_size_mb > 100:  # 100MB threshold
                st.warning("⚠️ Large file detected. Processing may take time.")
            
            if file_size_mb > 1000:  # 1GB threshold
                st.error("🚨 Very large file detected. Consider using database import or large file processing.")
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.info("Please ensure the file is a valid CSV format.")
    
    else:
        # Clear file info when no file is uploaded
        if "uploaded_file" in st.session_state:
            del st.session_state.uploaded_file
        set_file_info({})
    
    # File upload tips
    with st.expander("💡 File Upload Tips"):
        st.markdown("""
        **Supported Formats:**
        - CSV files (.csv)
        - UTF-8 encoding recommended
        - Maximum file size: 3GB+
        
        **Best Practices:**
        - Use clear column headers
        - Avoid special characters in column names
        - Ensure consistent data types
        - Remove unnecessary columns before upload
        
        **Large Files:**
        - Files > 100MB may take longer to process
        - Consider using database import for very large datasets
        - Turbo mode is recommended for large files
        """)

def render_large_file_upload(mode: str = "fast"):
    """Render large file upload component with special handling"""
    
    st.markdown("#### 📁 Large File Upload")
    
    # Large file uploader
    uploaded_file = st.file_uploader(
        "Choose a large CSV file",
        type=['csv'],
        key=f"{mode}_large_file_uploader",
        help="Upload a large CSV file (>100MB) for processing"
    )
    
    if uploaded_file is not None:
        # Store uploaded file in session state
        st.session_state.uploaded_file = uploaded_file
        
        # Display file information
        file_info = {
            "filename": uploaded_file.name,
            "size": uploaded_file.size,
            "type": uploaded_file.type,
            "is_large": True
        }
        
        set_file_info(file_info)
        
        # Show file details
        file_size_mb = uploaded_file.size / (1024 * 1024)
        file_size_gb = file_size_mb / 1024
        
        st.markdown(f"**File:** {uploaded_file.name}")
        st.markdown(f"**Size:** {file_size_gb:.2f} GB ({file_size_mb:.2f} MB)")
        
        # Large file processing options
        st.markdown("#### ⚙️ Large File Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.use_turbo = st.checkbox(
                "Enable Turbo Mode", 
                value=True, 
                key=f"{mode}_turbo_large",
                help="Faster processing for large files"
            )
            
            st.session_state.batch_size = st.slider(
                "Batch Size",
                min_value=64,
                max_value=512,
                value=256,
                key=f"{mode}_batch_size_large",
                help="Larger batches for better performance"
            )
        
        with col2:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=200,
                max_value=800,
                value=400,
                key=f"{mode}_chunk_size_large",
                help="Larger chunks for better context"
            )
            
            overlap = st.slider(
                "Overlap",
                min_value=0,
                max_value=100,
                value=50,
                key=f"{mode}_overlap_large",
                help="Overlap between chunks"
            )
        
        # Processing estimate
        estimated_time = estimate_processing_time(file_size_mb)
        st.info(f"⏱️ Estimated processing time: {estimated_time}")
        
        # Memory warning
        if file_size_mb > 500:  # 500MB threshold
            st.warning("⚠️ Very large file detected. Ensure sufficient system memory.")
        
        if file_size_mb > 2000:  # 2GB threshold
            st.error("🚨 Extremely large file detected. Consider using database import instead.")
    
    else:
        # Clear file info when no file is uploaded
        if "uploaded_file" in st.session_state:
            del st.session_state.uploaded_file
        set_file_info({})

def estimate_processing_time(file_size_mb: float) -> str:
    """Estimate processing time based on file size"""
    if file_size_mb < 10:
        return "1-2 minutes"
    elif file_size_mb < 100:
        return "5-10 minutes"
    elif file_size_mb < 500:
        return "15-30 minutes"
    elif file_size_mb < 1000:
        return "30-60 minutes"
    else:
        return "1-2 hours"
