# Processing Configuration Components
import streamlit as st
from ..utils.styling import render_info_card

def render_processing_config(mode: str = "fast"):
    """Render processing configuration component"""
    
    st.markdown("#### ⚙️ Processing Configuration")
    
    if mode == "fast":
        render_fast_config()
    elif mode == "config1":
        render_config1_config()
    else:
        st.info("Configuration options will be available based on selected mode")

def render_fast_config():
    """Render Fast Mode configuration"""
    
    # Fast mode options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🚀 Performance Options")
        
        use_turbo = st.checkbox(
            "Enable Turbo Mode",
            value=True,
            help="Faster processing with optimized algorithms"
        )
        
        batch_size = st.slider(
            "Batch Size",
            min_value=64,
            max_value=512,
            value=256,
            help="Batch size for embedding generation"
        )
        
        process_large_files = st.checkbox(
            "Process Large Files",
            value=True,
            help="Enable large file processing capabilities"
        )
    
    with col2:
        st.markdown("##### 🔧 Processing Options")
        
        apply_default_preprocessing = st.checkbox(
            "Apply Default Preprocessing",
            value=True,
            help="Apply automatic text cleaning and normalization"
        )
        
        use_openai = st.checkbox(
            "Use OpenAI API",
            value=False,
            help="Use OpenAI's embedding API instead of local models"
        )
        
        if use_openai:
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Your OpenAI API key"
            )
            openai_base_url = st.text_input(
                "OpenAI Base URL (optional)",
                help="Custom OpenAI API base URL"
            )
    
    # Fast mode info
    with st.expander("ℹ️ Fast Mode Information"):
        st.markdown("""
        **Fast Mode Features:**
        - Automatic semantic clustering
        - Optimized preprocessing pipeline
        - Default paraphrase-MiniLM-L6-v2 model
        - FAISS storage for fast retrieval
        - Turbo mode for large files
        
        **Best For:**
        - Quick processing with good results
        - Large datasets (>10MB)
        - When you need fast results
        - General-purpose text processing
        
        **Processing Steps:**
        1. Automatic preprocessing
        2. Semantic clustering chunking
        3. Embedding generation
        4. FAISS storage
        5. Retrieval system ready
        """)

def render_config1_config():
    """Render Config-1 Mode configuration"""
    
    # Chunking configuration
    st.markdown("##### ✂️ Chunking Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_method = st.selectbox(
            "Chunking Method",
            ["fixed", "recursive", "semantic", "document"],
            help="Select the chunking algorithm"
        )
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=1000,
            value=400,
            help="Maximum characters per chunk"
        )
        
        overlap = st.slider(
            "Overlap",
            min_value=0,
            max_value=200,
            value=50,
            help="Character overlap between chunks"
        )
    
    with col2:
        if chunk_method == "document":
            document_key_column = st.text_input(
                "Key Column",
                help="Column to group by for document chunking"
            )
        else:
            document_key_column = None
        
        token_limit = st.slider(
            "Token Limit",
            min_value=500,
            max_value=5000,
            value=2000,
            help="Maximum tokens per chunk"
        )
        
        preserve_headers = st.checkbox(
            "Preserve Headers",
            value=True,
            help="Include column headers in chunks"
        )
    
    # Model and storage configuration
    st.markdown("##### 🧠 Model & Storage Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox(
            "Embedding Model",
            ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L6-v2", "text-embedding-ada-002"],
            help="Select the embedding model"
        )
        
        storage_choice = st.selectbox(
            "Storage Backend",
            ["faiss", "chroma"],
            help="Select the vector storage backend"
        )
        
        retrieval_metric = st.selectbox(
            "Retrieval Metric",
            ["cosine", "dot", "euclidean"],
            help="Similarity metric for retrieval"
        )
    
    with col2:
        batch_size = st.slider(
            "Batch Size",
            min_value=64,
            max_value=512,
            value=256,
            help="Batch size for embedding generation"
        )
        
        use_parallel = st.checkbox(
            "Use Parallel Processing",
            value=True,
            help="Enable parallel processing for faster embedding"
        )
        
        apply_default_preprocessing = st.checkbox(
            "Apply Default Preprocessing",
            value=True,
            help="Apply automatic text cleaning and normalization"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            use_openai = st.checkbox(
                "Use OpenAI API",
                value=False,
                help="Use OpenAI's embedding API"
            )
            
            if use_openai:
                openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Your OpenAI API key"
                )
                openai_base_url = st.text_input(
                    "OpenAI Base URL (optional)",
                    help="Custom OpenAI API base URL"
                )
            
            use_turbo = st.checkbox(
                "Enable Turbo Mode",
                value=False,
                help="Faster processing with optimized algorithms"
            )
        
        with col2:
            process_large_files = st.checkbox(
                "Process Large Files",
                value=True,
                help="Enable large file processing capabilities"
            )
            
            # Performance estimate
            estimated_time = estimate_processing_time(chunk_size, batch_size)
            st.info(f"⏱️ Estimated processing time: {estimated_time}")
    
    # Config-1 mode info
    with st.expander("ℹ️ Config-1 Mode Information"):
        st.markdown("""
        **Config-1 Mode Features:**
        - 4 chunking methods: Fixed, Recursive, Semantic, Document
        - Multiple embedding models
        - FAISS or ChromaDB storage
        - Configurable retrieval metrics
        - Advanced preprocessing options
        
        **Best For:**
        - Balanced control and performance
        - Specific chunking requirements
        - Custom embedding models
        - Production environments
        
        **Chunking Methods:**
        - **Fixed**: Fixed-size chunks with overlap
        - **Recursive**: Recursive character splitting
        - **Semantic**: Clustering-based chunking
        - **Document**: Group by key column
        
        **Storage Options:**
        - **FAISS**: Fast similarity search
        - **ChromaDB**: Persistent vector storage
        """)

def estimate_processing_time(chunk_size: int, batch_size: int) -> str:
    """Estimate processing time based on configuration"""
    # Simple estimation based on chunk size and batch size
    if chunk_size < 300 and batch_size > 200:
        return "2-5 minutes"
    elif chunk_size < 500 and batch_size > 100:
        return "5-10 minutes"
    elif chunk_size < 700 and batch_size > 64:
        return "10-20 minutes"
    else:
        return "20-30 minutes"

def render_model_info(model_name: str):
    """Render model information"""
    
    model_info = {
        "paraphrase-MiniLM-L6-v2": {
            "description": "Fast and efficient sentence transformer",
            "dimension": 384,
            "speed": "Fast",
            "quality": "Good",
            "use_case": "General purpose, fast processing"
        },
        "all-MiniLM-L6-v2": {
            "description": "Versatile sentence transformer",
            "dimension": 384,
            "speed": "Fast",
            "quality": "Good",
            "use_case": "General purpose, balanced performance"
        },
        "text-embedding-ada-002": {
            "description": "OpenAI's high-quality embedding model",
            "dimension": 1536,
            "speed": "Medium",
            "quality": "Excellent",
            "use_case": "High-quality embeddings, OpenAI API required"
        }
    }
    
    info = model_info.get(model_name, {})
    
    if info:
        st.markdown(f"##### 📊 {model_name} Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(render_info_card(
                "Description",
                info.get("description", "N/A")
            ), unsafe_allow_html=True)
            
            st.markdown(render_info_card(
                "Dimension",
                str(info.get("dimension", "N/A"))
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(render_info_card(
                "Speed",
                info.get("speed", "N/A")
            ), unsafe_allow_html=True)
            
            st.markdown(render_info_card(
                "Quality",
                info.get("quality", "N/A")
            ), unsafe_allow_html=True)
        
        st.markdown(render_info_card(
            "Use Case",
            info.get("use_case", "N/A")
        ), unsafe_allow_html=True)

def render_storage_info(storage_type: str):
    """Render storage information"""
    
    storage_info = {
        "faiss": {
            "description": "Facebook AI Similarity Search",
            "speed": "Very Fast",
            "memory": "In-memory",
            "persistence": "Temporary",
            "use_case": "Fast similarity search, temporary storage"
        },
        "chroma": {
            "description": "ChromaDB vector database",
            "speed": "Fast",
            "memory": "Disk-based",
            "persistence": "Persistent",
            "use_case": "Persistent storage, production use"
        }
    }
    
    info = storage_info.get(storage_type, {})
    
    if info:
        st.markdown(f"##### 💾 {storage_type.upper()} Storage Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(render_info_card(
                "Description",
                info.get("description", "N/A")
            ), unsafe_allow_html=True)
            
            st.markdown(render_info_card(
                "Speed",
                info.get("speed", "N/A")
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(render_info_card(
                "Memory Usage",
                info.get("memory", "N/A")
            ), unsafe_allow_html=True)
            
            st.markdown(render_info_card(
                "Persistence",
                info.get("persistence", "N/A")
            ), unsafe_allow_html=True)
        
        st.markdown(render_info_card(
            "Use Case",
            info.get("use_case", "N/A")
        ), unsafe_allow_html=True)
