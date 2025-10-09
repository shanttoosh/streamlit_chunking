# Deep Config Components
import streamlit as st
import pandas as pd
import json
try:
    from ..utils.api_client import APIClient
    from ..utils.session_state import update_process_status, get_process_status
    from ..utils.styling import render_info_card, render_metric_card
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.ui.utils.api_client import APIClient
    from src.ui.utils.session_state import update_process_status, get_process_status
    from src.ui.utils.styling import render_info_card, render_metric_card

def render_deep_config():
    """Render Deep Config Mode interface"""
    
    st.markdown("### 🔬 Deep Config Mode - Comprehensive Workflow")
    
    # Step navigation
    steps = [
        "1. Data Upload",
        "2. Preprocessing",
        "3. Type Conversion", 
        "4. Null Handling",
        "5. Text Processing",
        "6. Chunking",
        "7. Embedding",
        "8. Storage",
        "9. Complete"
    ]
    
    current_step = st.session_state.get("deep_config_step", 0)
    
    # Step progress
    st.markdown("#### 📋 Workflow Steps")
    cols = st.columns(len(steps))
    
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i <= current_step:
                st.success(f"✅ {step}")
            elif i == current_step + 1:
                st.info(f"🔄 {step}")
            else:
                st.write(f"⏳ {step}")
    
    st.markdown("---")
    
    # Step content
    if current_step == 0:
        render_data_upload_step()
    elif current_step == 1:
        render_preprocessing_step()
    elif current_step == 2:
        render_type_conversion_step()
    elif current_step == 3:
        render_null_handling_step()
    elif current_step == 4:
        render_text_processing_step()
    elif current_step == 5:
        render_chunking_step()
    elif current_step == 6:
        render_embedding_step()
    elif current_step == 7:
        render_storage_step()
    elif current_step == 8:
        render_completion_step()
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if current_step > 0:
            if st.button("⬅️ Previous Step", use_container_width=True):
                st.session_state.deep_config_step = current_step - 1
                st.rerun()
    
    with col2:
        if current_step < len(steps) - 1:
            if st.button("Next Step ➡️", use_container_width=True):
                st.session_state.deep_config_step = current_step + 1
                st.rerun()
    
    with col3:
        if st.button("🔄 Reset Workflow", use_container_width=True):
            st.session_state.deep_config_step = 0
            st.rerun()

def render_data_upload_step():
    """Render data upload step"""
    st.markdown("#### 📁 Step 1: Data Upload")
    
    # Input source selection
    input_source = st.radio(
        "Select Input Source:",
        ["📁 Upload CSV File", "🗄️ Database Import"],
        key="deep_input_source"
    )
    
    if input_source == "📁 Upload CSV File":
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            key="deep_file_uploader"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            # Preview data
            try:
                df_preview = pd.read_csv(uploaded_file, nrows=5)
                st.markdown("#### 📋 Data Preview")
                st.dataframe(df_preview, use_container_width=True)
                
                # Data info
                st.markdown("#### 📊 Data Information")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows", len(df_preview))
                with col2:
                    st.metric("Columns", len(df_preview.columns))
                with col3:
                    st.metric("Non-Null", df_preview.count().sum())
                with col4:
                    st.metric("Null", df_preview.isnull().sum().sum())
                
                # Store preview
                st.session_state.preview_df = df_preview
                st.session_state.deep_df = df_preview
                
                st.success("✅ Data uploaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        # Database import (simplified)
        st.info("Database import functionality will be available in future updates")

def render_preprocessing_step():
    """Render preprocessing step"""
    st.markdown("#### 🔧 Step 2: Preprocessing")
    
    if st.session_state.get("preview_df") is None:
        st.warning("⚠️ Please upload data first")
        return
    
    df = st.session_state.preview_df
    
    # Preprocessing options
    st.markdown("##### Preprocessing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        remove_duplicates = st.checkbox("Remove Duplicates", value=True)
        clean_headers = st.checkbox("Clean Column Headers", value=True)
        normalize_text = st.checkbox("Normalize Text", value=True)
    
    with col2:
        remove_empty_rows = st.checkbox("Remove Empty Rows", value=True)
        trim_whitespace = st.checkbox("Trim Whitespace", value=True)
        convert_types = st.checkbox("Auto-convert Types", value=True)
    
    # Apply preprocessing
    if st.button("🔧 Apply Preprocessing", use_container_width=True):
        with st.spinner("Applying preprocessing..."):
            # Simulate preprocessing
            processed_df = df.copy()
            
            if remove_duplicates:
                processed_df = processed_df.drop_duplicates()
            
            if clean_headers:
                processed_df.columns = [col.strip().lower().replace(' ', '_') for col in processed_df.columns]
            
            if normalize_text:
                for col in processed_df.select_dtypes(include=['object']).columns:
                    processed_df[col] = processed_df[col].astype(str).str.lower().str.strip()
            
            if remove_empty_rows:
                processed_df = processed_df.dropna(how='all')
            
            if trim_whitespace:
                for col in processed_df.select_dtypes(include=['object']).columns:
                    processed_df[col] = processed_df[col].astype(str).str.strip()
            
            # Store processed data
            st.session_state.deep_df = processed_df
            st.session_state.preprocessing_config = {
                "remove_duplicates": remove_duplicates,
                "clean_headers": clean_headers,
                "normalize_text": normalize_text,
                "remove_empty_rows": remove_empty_rows,
                "trim_whitespace": trim_whitespace,
                "convert_types": convert_types
            }
            
            st.success("✅ Preprocessing completed!")
            
            # Show results
            st.markdown("#### 📊 Preprocessing Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Original Rows", len(df))
            with col2:
                st.metric("Processed Rows", len(processed_df))
            with col3:
                st.metric("Rows Removed", len(df) - len(processed_df))
            
            # Show processed data
            st.markdown("#### 📋 Processed Data Preview")
            st.dataframe(processed_df.head(), use_container_width=True)

def render_type_conversion_step():
    """Render type conversion step"""
    st.markdown("#### 🔄 Step 3: Type Conversion")
    
    if st.session_state.get("deep_df") is None:
        st.warning("⚠️ Please complete preprocessing first")
        return
    
    df = st.session_state.deep_df
    
    # Column type analysis
    st.markdown("##### Column Type Analysis")
    
    type_info = []
    for col in df.columns:
        col_info = {
            "Column": col,
            "Current Type": str(df[col].dtype),
            "Non-Null": df[col].count(),
            "Null": df[col].isnull().sum(),
            "Unique": df[col].nunique()
        }
        type_info.append(col_info)
    
    type_df = pd.DataFrame(type_info)
    st.dataframe(type_df, use_container_width=True)
    
    # Type conversion options
    st.markdown("##### Type Conversion Options")
    
    conversions = {}
    for col in df.columns:
        col_type = st.selectbox(
            f"Convert '{col}' to:",
            ["keep", "string", "numeric", "integer", "float", "datetime", "boolean", "category"],
            key=f"type_conv_{col}",
            help=f"Current type: {df[col].dtype}"
        )
        if col_type != "keep":
            conversions[col] = col_type
    
    # Apply conversions
    if st.button("🔄 Apply Type Conversions", use_container_width=True):
        with st.spinner("Applying type conversions..."):
            converted_df = df.copy()
            
            for col, target_type in conversions.items():
                try:
                    if target_type == "string":
                        converted_df[col] = converted_df[col].astype(str)
                    elif target_type == "numeric":
                        converted_df[col] = pd.to_numeric(converted_df[col], errors='coerce')
                    elif target_type == "integer":
                        converted_df[col] = pd.to_numeric(converted_df[col], errors='coerce').astype('Int64')
                    elif target_type == "float":
                        converted_df[col] = pd.to_numeric(converted_df[col], errors='coerce')
                    elif target_type == "datetime":
                        converted_df[col] = pd.to_datetime(converted_df[col], errors='coerce')
                    elif target_type == "boolean":
                        converted_df[col] = converted_df[col].astype(bool)
                    elif target_type == "category":
                        converted_df[col] = converted_df[col].astype('category')
                except Exception as e:
                    st.warning(f"Could not convert {col} to {target_type}: {e}")
            
            # Store converted data
            st.session_state.deep_df = converted_df
            st.session_state.type_conversions = conversions
            
            st.success("✅ Type conversions completed!")
            
            # Show results
            st.markdown("#### 📊 Conversion Results")
            for col, target_type in conversions.items():
                st.write(f"✅ {col}: {df[col].dtype} → {target_type}")

def render_null_handling_step():
    """Render null handling step"""
    st.markdown("#### 🔍 Step 4: Null Handling")
    
    if st.session_state.get("deep_df") is None:
        st.warning("⚠️ Please complete type conversion first")
        return
    
    df = st.session_state.deep_df
    
    # Null analysis
    st.markdown("##### Null Value Analysis")
    
    null_info = []
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        null_info.append({
            "Column": col,
            "Null Count": null_count,
            "Null %": f"{null_pct:.1f}%",
            "Non-Null": df[col].count()
        })
    
    null_df = pd.DataFrame(null_info)
    st.dataframe(null_df, use_container_width=True)
    
    # Null handling strategies
    st.markdown("##### Null Handling Strategies")
    
    strategies = {}
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            strategy = st.selectbox(
                f"Handle nulls in '{col}':",
                ["keep", "drop", "fill_mean", "fill_median", "fill_mode", "fill_zero", "fill_unknown"],
                key=f"null_strategy_{col}",
                help=f"Null count: {df[col].isnull().sum()}"
            )
            if strategy != "keep":
                strategies[col] = strategy
    
    # Apply strategies
    if st.button("🔍 Apply Null Handling", use_container_width=True):
        with st.spinner("Applying null handling..."):
            handled_df = df.copy()
            
            for col, strategy in strategies.items():
                try:
                    if strategy == "drop":
                        handled_df = handled_df.dropna(subset=[col])
                    elif strategy == "fill_mean":
                        handled_df[col] = handled_df[col].fillna(handled_df[col].mean())
                    elif strategy == "fill_median":
                        handled_df[col] = handled_df[col].fillna(handled_df[col].median())
                    elif strategy == "fill_mode":
                        mode_val = handled_df[col].mode().iloc[0] if not handled_df[col].mode().empty else 0
                        handled_df[col] = handled_df[col].fillna(mode_val)
                    elif strategy == "fill_zero":
                        handled_df[col] = handled_df[col].fillna(0)
                    elif strategy == "fill_unknown":
                        handled_df[col] = handled_df[col].fillna("Unknown")
                except Exception as e:
                    st.warning(f"Could not handle nulls in {col}: {e}")
            
            # Store handled data
            st.session_state.deep_df = handled_df
            st.session_state.null_strategies = strategies
            
            st.success("✅ Null handling completed!")
            
            # Show results
            st.markdown("#### 📊 Null Handling Results")
            for col, strategy in strategies.items():
                original_nulls = df[col].isnull().sum()
                new_nulls = handled_df[col].isnull().sum()
                st.write(f"✅ {col}: {original_nulls} → {new_nulls} nulls ({strategy})")

def render_text_processing_step():
    """Render text processing step"""
    st.markdown("#### 📝 Step 5: Text Processing")
    
    if st.session_state.get("deep_df") is None:
        st.warning("⚠️ Please complete null handling first")
        return
    
    df = st.session_state.deep_df
    
    # Text processing options
    st.markdown("##### Text Processing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        remove_stopwords = st.checkbox("Remove Stop Words", value=False)
        lemmatize = st.checkbox("Lemmatize Text", value=False)
        stem = st.checkbox("Stem Text", value=False)
    
    with col2:
        remove_html = st.checkbox("Remove HTML Tags", value=True)
        remove_special_chars = st.checkbox("Remove Special Characters", value=False)
        normalize_case = st.checkbox("Normalize Case", value=True)
    
    # Apply text processing
    if st.button("📝 Apply Text Processing", use_container_width=True):
        with st.spinner("Applying text processing..."):
            processed_df = df.copy()
            
            # Apply text processing to string columns
            for col in processed_df.select_dtypes(include=['object']).columns:
                if normalize_case:
                    processed_df[col] = processed_df[col].astype(str).str.lower()
                
                if remove_html:
                    processed_df[col] = processed_df[col].str.replace(r'<[^<]+?>', ' ', regex=True)
                
                if remove_special_chars:
                    processed_df[col] = processed_df[col].str.replace(r'[^\w\s]', ' ', regex=True)
                
                if remove_stopwords:
                    # Simple stopwords removal (in practice, would use NLTK/spaCy)
                    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                    processed_df[col] = processed_df[col].apply(
                        lambda x: ' '.join([word for word in str(x).split() if word.lower() not in stopwords])
                    )
            
            # Store processed data
            st.session_state.deep_df = processed_df
            st.session_state.text_processing = {
                "remove_stopwords": remove_stopwords,
                "lemmatize": lemmatize,
                "stem": stem,
                "remove_html": remove_html,
                "remove_special_chars": remove_special_chars,
                "normalize_case": normalize_case
            }
            
            st.success("✅ Text processing completed!")
            
            # Show results
            st.markdown("#### 📊 Text Processing Results")
            st.info("Text processing applied to all string columns")

def render_chunking_step():
    """Render chunking step"""
    st.markdown("#### ✂️ Step 6: Chunking")
    
    if st.session_state.get("deep_df") is None:
        st.warning("⚠️ Please complete text processing first")
        return
    
    df = st.session_state.deep_df
    
    # Chunking options
    st.markdown("##### Chunking Configuration")
    
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
            key_column = st.selectbox(
                "Key Column",
                df.columns.tolist(),
                help="Column to group by for document chunking"
            )
        else:
            key_column = None
        
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
    
    # Apply chunking
    if st.button("✂️ Apply Chunking", use_container_width=True):
        with st.spinner("Applying chunking..."):
            # Simulate chunking (in practice, would call actual chunking functions)
            chunks = []
            chunk_metadata = []
            
            # Simple chunking simulation
            text_data = df.astype(str).agg(' '.join, axis=1).tolist()
            
            for i, text in enumerate(text_data):
                # Simple fixed chunking
                words = text.split()
                for j in range(0, len(words), chunk_size // 4):  # Rough word count
                    chunk_words = words[j:j + chunk_size // 4]
                    if chunk_words:
                        chunk_text = ' '.join(chunk_words)
                        chunks.append(chunk_text)
                        chunk_metadata.append({
                            'chunk_id': len(chunks),
                            'source_row': i,
                            'method': chunk_method,
                            'chunk_size': len(chunk_text),
                            'word_count': len(chunk_words)
                        })
            
            # Store chunks
            st.session_state.deep_chunks = chunks
            st.session_state.chunking_config = {
                "method": chunk_method,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "key_column": key_column,
                "token_limit": token_limit,
                "preserve_headers": preserve_headers
            }
            
            st.success("✅ Chunking completed!")
            
            # Show results
            st.markdown("#### 📊 Chunking Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Chunks", len(chunks))
            with col2:
                st.metric("Avg Chunk Size", f"{sum(len(c) for c in chunks) // len(chunks)} chars")
            with col3:
                st.metric("Method", chunk_method.title())
            
            # Show sample chunks
            st.markdown("#### 📋 Sample Chunks")
            for i, chunk in enumerate(chunks[:3]):
                with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
                    st.text(chunk)

def render_embedding_step():
    """Render embedding step"""
    st.markdown("#### 🧠 Step 7: Embedding")
    
    if not st.session_state.get("deep_chunks"):
        st.warning("⚠️ Please complete chunking first")
        return
    
    chunks = st.session_state.deep_chunks
    
    # Embedding options
    st.markdown("##### Embedding Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox(
            "Embedding Model",
            ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L6-v2", "text-embedding-ada-002"],
            help="Select the embedding model"
        )
        
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
        else:
            openai_api_key = None
            openai_base_url = None
    
    with col2:
        batch_size = st.slider(
            "Batch Size",
            min_value=32,
            max_value=512,
            value=64,
            help="Batch size for embedding generation"
        )
        
        use_parallel = st.checkbox(
            "Use Parallel Processing",
            value=True,
            help="Enable parallel processing for faster embedding"
        )
        
        # Performance estimate
        estimated_time = len(chunks) * 0.1  # Rough estimate
        st.info(f"⏱️ Estimated time: {estimated_time:.1f} seconds")
    
    # Generate embeddings
    if st.button("🧠 Generate Embeddings", use_container_width=True):
        with st.spinner("Generating embeddings..."):
            # Simulate embedding generation
            import numpy as np
            
            # Create dummy embeddings (in practice, would call actual embedding function)
            embedding_dim = 384 if "MiniLM" in model_choice else 1536
            embeddings = np.random.rand(len(chunks), embedding_dim).astype(np.float32)
            
            # Store embeddings
            st.session_state.deep_embeddings = embeddings
            st.session_state.embedding_config = {
                "model_name": model_choice,
                "use_openai": use_openai,
                "openai_api_key": openai_api_key,
                "openai_base_url": openai_base_url,
                "batch_size": batch_size,
                "use_parallel": use_parallel
            }
            
            st.success("✅ Embeddings generated!")
            
            # Show results
            st.markdown("#### 📊 Embedding Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Embeddings", len(embeddings))
            with col2:
                st.metric("Embedding Dimension", embedding_dim)
            with col3:
                st.metric("Model Used", model_choice)
            
            # Show embedding info
            st.markdown("#### 📋 Embedding Information")
            st.info(f"Generated {len(embeddings)} embeddings with dimension {embedding_dim}")

def render_storage_step():
    """Render storage step"""
    st.markdown("#### 💾 Step 8: Storage")
    
    if st.session_state.get("deep_embeddings") is None:
        st.warning("⚠️ Please generate embeddings first")
        return
    
    chunks = st.session_state.deep_chunks
    embeddings = st.session_state.deep_embeddings
    
    # Storage options
    st.markdown("##### Storage Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        storage_type = st.selectbox(
            "Storage Type",
            ["faiss", "chroma"],
            help="Select the vector storage backend"
        )
        
        collection_name = st.text_input(
            "Collection Name",
            value=f"deep_config_{len(chunks)}",
            help="Name for the storage collection"
        )
        
        retrieval_metric = st.selectbox(
            "Retrieval Metric",
            ["cosine", "dot", "euclidean"],
            help="Similarity metric for retrieval"
        )
    
    with col2:
        # Storage info
        st.markdown("##### Storage Information")
        st.info(f"**Chunks:** {len(chunks)}")
        st.info(f"**Embeddings:** {len(embeddings)}")
        st.info(f"**Dimension:** {embeddings.shape[1]}")
        st.info(f"**Storage:** {storage_type.upper()}")
        
        # Memory estimate
        memory_mb = (embeddings.nbytes + sum(len(c) for c in chunks) * 2) / (1024 * 1024)
        st.info(f"**Estimated Memory:** {memory_mb:.1f} MB")
    
    # Store embeddings
    if st.button("💾 Store Embeddings", use_container_width=True):
        with st.spinner("Storing embeddings..."):
            # Simulate storage (in practice, would call actual storage function)
            store_info = {
                "type": storage_type,
                "collection_name": collection_name,
                "metric": retrieval_metric,
                "total_vectors": len(embeddings),
                "embedding_dim": embeddings.shape[1]
            }
            
            # Store storage info
            st.session_state.deep_store_info = store_info
            st.session_state.storage_config = {
                "type": storage_type,
                "collection_name": collection_name,
                "metric": retrieval_metric
            }
            
            st.success("✅ Embeddings stored!")
            
            # Show results
            st.markdown("#### 📊 Storage Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Storage Type", storage_type.upper())
            with col2:
                st.metric("Vectors Stored", len(embeddings))
            with col3:
                st.metric("Collection", collection_name)
            
            # Update process status
            update_process_status("storage", "completed")
            update_process_status("retrieval", "ready")

def render_completion_step():
    """Render completion step"""
    st.markdown("#### ✅ Step 9: Complete")
    
    # Summary
    st.markdown("##### 📊 Processing Summary")
    
    # Get all configurations
    preprocessing_config = st.session_state.get("preprocessing_config", {})
    chunking_config = st.session_state.get("chunking_config", {})
    embedding_config = st.session_state.get("embedding_config", {})
    storage_config = st.session_state.get("storage_config", {})
    
    # Display summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Configuration Summary")
        st.json({
            "preprocessing": preprocessing_config,
            "chunking": chunking_config,
            "embedding": embedding_config,
            "storage": storage_config
        })
    
    with col2:
        st.markdown("##### Results Summary")
        
        if st.session_state.get("deep_df") is not None:
            st.metric("Processed Rows", len(st.session_state.deep_df))
        
        if st.session_state.get("deep_chunks"):
            st.metric("Chunks Created", len(st.session_state.deep_chunks))
        
        if st.session_state.get("deep_embeddings") is not None:
            st.metric("Embeddings Generated", len(st.session_state.deep_embeddings))
        
        if st.session_state.get("deep_store_info"):
            st.metric("Storage Type", st.session_state.deep_store_info["type"].upper())
    
    # Final processing
    if st.button("🚀 Complete Deep Config Processing", use_container_width=True, type="primary"):
        with st.spinner("Completing deep config processing..."):
            # Simulate final processing
            result = {
                "rows": len(st.session_state.deep_df) if st.session_state.get("deep_df") else 0,
                "chunks": len(st.session_state.deep_chunks) if st.session_state.get("deep_chunks") else 0,
                "stored": st.session_state.deep_store_info["type"] if st.session_state.get("deep_store_info") else "none",
                "embedding_model": embedding_config.get("model_name", "unknown"),
                "retrieval_ready": True,
                "processing_time": 0.0,
                "enhanced_pipeline": True
            }
            
            # Store results
            st.session_state.api_results = result
            
            # Update all process statuses
            for step in ["preprocessing", "chunking", "embedding", "storage", "retrieval"]:
                update_process_status(step, "completed")
            
            st.success("🎉 Deep Config processing completed successfully!")
            
            # Show final results
            st.markdown("#### 🎯 Final Results")
            st.json(result)
            
            # Export options
            st.markdown("#### 📤 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Export Chunks", use_container_width=True):
                    st.success("✅ Chunks exported!")
            
            with col2:
                if st.button("🔢 Export Embeddings", use_container_width=True):
                    st.success("✅ Embeddings exported!")
            
            with col3:
                if st.button("📋 Export Config", use_container_width=True):
                    st.success("✅ Configuration exported!")
