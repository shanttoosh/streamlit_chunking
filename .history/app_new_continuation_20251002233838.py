            # Process form submission
            if test_connection or list_tables:
                db_payload = {
                    "db_type": db_type,
                    "host": host,
                    "port": port,
                    "username": username,
                    "password": password,
                    "database": database,
                }
                
                # Test connection
                if test_connection:
                    with st.spinner("Testing connection..."):
                        try:
                            result = db_test_connection_api(db_payload)
                            if result.get("status") == "success":
                                st.success("✅ **Connection Successful!**")
                                st.session_state.db_credentials = db_payload
                            else:
                                st.error(f"❌ **Connection Failed:** {result.get('message', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"❌ **Connection Error:** {str(e)}")
                
                # List tables
                if list_tables:
                    with st.spinner("Fetching tables..."):
                        try:
                            result = db_list_tables_api(db_payload)
                            if "error" in result:
                                st.error(f"❌ **Error:** {result['error']}")
                            else:
                                tables = result.get("tables", [])
                                if tables:
                                    st.success(f"✅ **Found {len(tables)} tables:**")
                                    st.session_state.db_tables = tables
                                    
                                    # Display tables
                                    for i, table in enumerate(tables, 1):
                                        st.write(f"{i}. **{table}**")
                                    
                                    # Table selection for processing
                                    st.markdown("---")
                                    st.markdown("#### Select Table for Processing")
                                    
                                    selected_table = st.selectbox(
                                        "Choose a table to process:",
                                        tables,
                                        help="Select the table you want to import and process"
                                    )
                                    
                                    if selected_table:
                                        st.session_state.selected_table = selected_table
                                        st.session_state.db_credentials = db_payload
                                        
                                        # Show table selection confirmation
                                        st.info(f"📋 **Table Selected:** {selected_table}")
                                        
                                        # Mode-specific processing controls
                                        st.markdown("#### Processing Options")
                                        
                                        # Fast Mode processing
                                        if st.session_state.selected_mode == "fast":
                                            if st.button("📥 Import & Process (Fast Mode)", key="db_fast_process", use_container_width=True):
                                                with st.spinner("Processing table with Fast Mode..."):
                                                    try:
                                                        payload = {
                                                            **db_payload,
                                                            "table_name": selected_table
                                                        }
                                                        result = db_import_one_api(payload)
                                                        
                                                        if "error" in result:
                                                            st.error(f"❌ **Processing Error:** {result['error']}")
                                                        else:
                                                            st.success("✅ **Table processed successfully!**")
                                                            st.session_state.api_results = result
                                                            # Update processing status
                                                            st.session_state.processing_status.update({
                                                                "preprocessing": True,
                                                                "chunking": True,
                                                                "embedding": True,
                                                                "storage": True,
                                                                "completed": True
                                                            })
                                                            
                                                            # Display results summary
                                                            if "summary" in result:
                                                                summary = result["summary"]
                                                                st.markdown("#### 📊 Processing Results")
                                                                st.json(summary)
                                                    except Exception as e:
                                                        st.error(f"❌ **Error:** {str(e)}")
                                        
                                        # Config-1 Mode processing
                                        elif st.session_state.selected_mode == "config1":
                                            with st.expander("⚙️ Config-1 Settings"):
                                                c1_col1, c1_col2 = st.columns(2)
                                                with c1_col1:
                                                    c1_null_handling = st.selectbox("Null Handling", ["keep", "drop", "fill"], key="db_c1_null")
                                                    c1_chunk_method = st.selectbox("Chunk Method", ["fixed", "recursive", "semantic", "document"], key="db_c1_chunk")
                                                    c1_model_choice = st.selectbox("Model", ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"], key="db_c1_model")
                                                with c1_col2:
                                                    c1_fill_value = st.text_input("Fill Value", "Unknown", key="db_c1_fill") if c1_null_handling == "fill" else None
                                                    c1_chunk_size = st.number_input("Chunk Size", 100, 2000, 400, key="db_c1_size")
                                                    c1_overlap = st.number_input("Overlap", 0, 200, 50, key="db_c1_overlap")
                                                    c1_storage_choice = st.selectbox("Storage", ["faiss", "chromadb"], key="db_c1_storage")
                                            
                                            if st.button("📥 Import & Process (Config-1 Mode)", key="db_config1_process", use_container_width=True):
                                                with st.spinner("Processing table with Config-1 Mode..."):
                                                    try:
                                                        payload = {
                                                            **db_payload,
                                                            "table_name": selected_table,
                                                            "null_handling": c1_null_handling,
                                                            "fill_value": c1_fill_value or "Unknown",
                                                            "chunk_method": c1_chunk_method,
                                                            "chunk_size": c1_chunk_size,
                                                            "overlap": c1_overlap,
                                                            "model_choice": c1_model_choice,
                                                            "storage_choice": c1_storage_choice,
                                                        }
                                                        response = requests.post(f"{API_BASE_URL}/db/run_config1", data=payload)
                                                        result = response.json()
                                                        
                                                        if "error" in result:
                                                            st.error(f"❌ **Processing Error:** {result['error']}")
                                                        else:
                                                            st.success("✅ **Table processed successfully!**")
                                                            st.session_state.api_results = result
                                                            # Update processing status
                                                            st.session_state.processing_status.update({
                                                                "preprocessing": True,
                                                                "chunking": True,
                                                                "embedding": True,
                                                                "storage": True,
                                                                "completed": True
                                                            })
                                                    except Exception as e:
                                                        st.error(f"❌ **Error:** {str(e)}")
                                        
                                        # Deep Mode processing
                                        elif st.session_state.selected_mode == "deep":
                                            with st.expander("🔬 Deep Mode Settings"):
                                                deep_col1, deep_col2 = st.columns(2)
                                                with deep_col1:
                                                    deep_null_handling = st.selectbox("Null Handling", ["keep", "drop", "fill"], key="db_deep_null")
                                                    deep_chunk_method = st.selectbox("Chunk Method", ["fixed", "recursive", "semantic", "document"], key="db_deep_chunk")
                                                    deep_model_choice = st.selectbox("Model", ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "sentence-transformers/all-mpnet-base-v2"], key="db_deep_model")
                                                    deep_remove_stopwords = st.checkbox("Remove Stopwords", key="db_deep_stopwords")
                                                    deep_lowercase = st.checkbox("Lowercase", value=True, key="db_deep_lowercase")
                                                with deep_col2:
                                                    deep_fill_value = st.text_input("Fill Value", "Unknown", key="db_deep_fill") if deep_null_handling == "fill" else None
                                                    deep_chunk_size = st.number_input("Chunk Size", 100, 2000, 400, key="db_deep_size")
                                                    deep_overlap = st.number_input("Overlap", 0, 200, 50, key="db_deep_overlap")
                                                    deep_storage_choice = st.selectbox("Storage", ["faiss", "chromadb"], key="db_deep_storage")
                                                    deep_stemming = st.checkbox("Stemming", key="db_deep_stemming")
                                                    deep_lemmatization = st.checkbox("Lemmatization", key="db_deep_lemmatization")
                                                  
                                            if st.button("📥 Import & Process (Deep Mode)", key="db_deep_process", use_container_width=True):
                                                with st.spinner("Processing table with Deep Mode..."):
                                                    try:
                                                        payload = {
                                                            **db_payload,
                                                            "table_name": selected_table,
                                                            "null_handling": deep_null_handling,
                                                            "fill_value": deep_fill_value or "Unknown",
                                                            "remove_stopwords": deep_remove_stopwords,
                                                            "lowercase": deep_lowercase,
                                                            "stemming": deep_stemming,
                                                            "lemmatization": deep_lemmatization,
                                                            "chunk_method": deep_chunk_method,
                                                            "chunk_size": deep_chunk_size,
                                                            "overlap": deep_overlap,
                                                            "model_choice": deep_model_choice,
                                                            "storage_choice": deep_storage_choice,
                                                        }
                                                        response = requests.post(f"{API_BASE_URL}/db/run_deep", data=payload)
                                                        result = response.json()
                                                        
                                                        if "error" in result:
                                                            st.error(f"❌ **Processing Error:** {result['error']}")
                                                        else:
                                                            st.success("✅ **Table processed successfully!**")
                                                            st.session_state.api_results = result
                                                            # Update processing status
                                                            st.session_state.processing_status.update({
                                                                "preprocessing": True,
                                                                "chunking": True,
                                                                "embedding": True,
                                                                "storage": True,
                                                                "completed": True
                                                            })
                                                    except Exception as e:
                                                        st.error(f"❌ **Error:** {str(e)}")
                                else:
                                    st.warning("⚠️ **No tables found in the database**")

# ---------- CSV Processing Pipeline ----------
if st.session_state.selected_mode and st.session_state.data_source == "csv" and hasattr(st.session_state, 'uploaded_file') and st.session_state.uploaded_file is not None:
    st.markdown("---")
    st.markdown("## 🚀 Processing Pipeline")
    
    # Mode-specific configuration based on selected mode
    if st.session_state.selected_mode == "fast":
        st.markdown("### ⚡ Fast Mode Configuration")
        st.markdown("""
        <div class="mode-card">
            <div class="mode-description">
                **Auto-optimized Pipeline**: This mode automatically handles preprocessing, 
                uses semantic clustering for chunking, MiniLM embeddings, and FAISS storage 
                for optimal performance with minimal configuration.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Run Fast Pipeline", key="csv_fast_process", use_container_width=True):
            with st.spinner("Running Fast Mode pipeline..."):
                try:
                    result = call_fast_api(
                        st.session_state.uploaded_file.getvalue(),
                        st.session_state.uploaded_file.name
                    )
                    
                    if "error" in result:
                        st.error(f"❌ **Processing Error:** {result['error']}")
                    else:
                        st.success("✅ **Fast pipeline completed successfully!**")
                        st.session_state.api_results = result
                        # Update processing status
                        st.session_state.processing_status.update({
                            "preprocessing": True,
                            "chunking": True,
                            "embedding": True,
                            "storage": True,
                            "completed": True
                        })
                        
                        # Display results summary
                        if "summary" in result:
                            summary = result["summary"]
                            st.markdown("#### 📊 Processing Results")
                            st.json(summary)
                            
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")
    
    elif st.session_state.selected_mode == "config1":
        st.markdown("### ⚙️ Config-1 Mode Configuration")
        
        with st.expander("⚙️ Config-1 Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🧹 Preprocessing")
                null_handling = st.selectbox(
                    "Null value handling",
                    ["keep", "drop", "fill"],
                    key="csv_config1_null_handling",
                    help="How to handle null values in the data"
                )
                fill_value = st.text_input("Fill value", "Unknown", key="csv_config1_fill") if null_handling == "fill" else None
                
                st.markdown("#### 📦 Chunking")
                chunk_method = st.selectbox(
                    "Chunking method",
                    ["fixed", "recursive", "semantic", "document"],
                    key="csv_config1_chunk_method",
                    help="Choose chunking strategy"
                )
                
                if chunk_method in ["fixed", "recursive"]:
                    chunk_size = st.number_input("Chunk size", 100, 2000, 400, key="csv_config1_chunk_size")
                    overlap = st.number_input("Overlap", 0, 500, 50, key="csv_config1_overlap")
            
            with col2:
                st.markdown("#### 🤖 Embedding")
                model_choice = st.selectbox(
                    "Embedding model",
                    ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"],
                    key="csv_config1_model_choice",
                    help="Choose embedding model"
                )
                
                st.markdown("#### 💾 Storage")
                storage_choice = st.selectbox(
                    "Vector storage",
                    ["faiss", "chromadb"],
                    key="csv_config1_storage_choice",
                    help="Choose vector storage backend"
                )
        
        if st.button("🚀 Run Config-1 Pipeline", key="csv_config1_process", use_container_width=True):
            with st.spinner("Running Config-1 pipeline..."):
                try:
                    config = {
                        "null_handling": null_handling,
                        "fill_value": fill_value or "Unknown",
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 400,
                        "overlap": overlap if 'overlap' in locals() else 50,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                    }
                    
                    result = call_config1_api(
                        st.session_state.uploaded_file.getvalue(),
                        st.session_state.uploaded_file.name,
                        config
                    )
                    
                    if "error" in result:
                        st.error(f"❌ **Processing Error:** {result['error']}")
                    else:
                        st.success("✅ **Config-1 pipeline completed successfully!**")
                        st.session_state.api_results = result
                        # Update processing status
                        st.session_state.processing_status.update({
                            "preprocessing": True,
                            "chunking": True,
                            "embedding": True,
                            "storage": True,
                            "completed": True
                        })
                        
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")
    
    elif st.session_state.selected_mode == "deep":
        st.markdown("### 🔬 Deep Mode Configuration")
        
        with st.expander("🔬 Deep Mode Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🧹 Preprocessing")
                null_handling = st.selectbox(
                    "Null value handling",
                    ["keep", "drop", "fill"],
                    key="csv_deep_null_handling",
                    help="How to handle null values"
                )
                fill_value = st.text_input("Fill value", "Unknown", key="csv_deep_fill") if null_handling == "fill" else None
                
                st.markdown("#### 🧠 Text Processing")
                remove_stopwords = st.checkbox("Remove stopwords", key="csv_deep_stopwords")
                lowercase = st.checkbox("Convert to lowercase", value=True, key="csv_deep_lowercase")
                stemming = st.checkbox("Apply stemming", key="csv_deep_stemming")
                lemmatization = st.checkbox("Apply lemmatization", key="csv_deep_lemmatization")
            
            with col2:
                st.markdown("#### 📦 Chunking")
                chunk_method = st.selectbox(
                    "Chunking method",
                    ["fixed", "recursive", "semantic", "document"],
                    key="csv_deep_chunk_method",
                    help="Advanced chunking strategy"
                )
                
                if chunk_method in ["fixed", "recursive"]:
                    chunk_size = st.number_input("Chunk size", 100, 2000, 400, key="csv_deep_chunk_size")
                    overlap = st.number_input("Overlap", 0, 500, 50, key="csv_deep_overlap")
                
                st.markdown("#### 🤖 Embedding & Storage")
                model_choice = st.selectbox(
                    "Embedding model",
                    ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"],
                    key="csv_deep_model_choice"
                )
                storage_choice = st.selectbox(
                    "Vector storage",
                    ["faiss", "chromadb"],
                    key="csv_deep_storage_choice"
                )
        
        if st.button("🚀 Run Deep Config Pipeline", key="csv_deep_process", use_container_width=True):
            with st.spinner("Running Deep Config pipeline..."):
                try:
                    config = {
                        "null_handling": null_handling,
                        "fill_value": fill_value or "Unknown",
                        "remove_stopwords": remove_stopwords,
                        "lowercase": lowercase,
                        "stemming": stemming,
                        "lemmatization": lemmatization,
                        "chunk_method": chunk_method,
                        "chunk_size": chunk_size if 'chunk_size' in locals() else 400,
                        "overlap": overlap if 'overlap' in locals() else 50,
                        "model_choice": model_choice,
                        "storage_choice": storage_choice,
                    }
                    
                    result = call_deep_api(
                        st.session_state.uploaded_file.getvalue(),
                        st.session_state.uploaded_file.name,
                        config
                    )
                    
                    if "error" in result:
                        st.error(f"❌ **Processing Error:** {result['error']}")
                    else:
                        st.success("✅ **Deep Config pipeline completed successfully!**")
                        st.session_state.api_results = result
                        # Update processing status
                        st.session_state.processing_status.update({
                            "preprocessing": True,
                            "chunking": True,
                            "embedding": True,
                            "storage": True,
                            "completed": True
                        })
                        
                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")

# ---------- Vector Retrieval Section ----------
if hasattr(st.session_state, 'api_results') and st.session_state.api_results and st.session_state.api_results.get('summary', {}).get('retrieval_ready'):
    st.markdown("---")
    st.markdown("## 🔍 Semantic Search")
    st.markdown("Search for similar content using semantic similarity")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        vector_query = st.text_input("Enter semantic search query:", placeholder="Search for similar content...", key="vector_query")
    with col2:
        k = st.slider("Top K results", 1, 10, 3, key="vector_k")
    
    if vector_query and st.button("🔍 Search", key="vector_search"):
        with st.spinner("Searching..."):
            try:
                retrieval_result = call_retrieve_api(vector_query, k)
                
                if "error" in retrieval_result:
                    st.error(f"❌ **Retrieval error:** {retrieval_result['error']}")
                else:
                    st.success(f"✅ **Found {len(retrieval_result['results'])} results**")
                    
                    for result in retrieval_result['results']:
                        similarity_color = "#28a745" if result['similarity'] > 0.7 else "#ffc107" if result['similarity'] > 0.4 else "#dc3545"
                        
                        st.markdown(f"""
                        <div style="background: #2d2d2d; border: 1px solid #444; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid {similarity_color};">
                            <h4 style="margin: 0 0 10px 0; color: {similarity_color};">
                                Rank #{result['rank']} (Similarity: {result['similarity']:.3f})
                            </h4>
                            <p style="margin: 0; color: #cccccc; font-size: 0.9em;">{result['content'][:300]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ **Error:** {str(e)}")

# ---------- Export Section ----------
if hasattr(st.session_state, 'api_results') and st.session_state.api_results:
    st.markdown("---")
    st.markdown("## 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📥 Download Chunks")
        if st.button("📄 Export Chunks as TXT", use_container_width=True):
            try:
                chunks_content = download_file("/export/chunks", "chunks.txt")
                st.download_button(
                    label="⬇️ Download Chunks",
                    data=chunks_content,
                    file_name="chunks.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ **Error exporting chunks:** {str(e)}")
    
    with col2:
        st.markdown("#### 📥 Download Embeddings")
        if st.button("🔢 Export Embeddings as NPY", use_container_width=True):
            try:
                embeddings_content = download_file("/export/embeddings", "embeddings.npy")
                st.download_button(
                    label="⬇️ Download Embeddings",
                    data=embeddings_content,
                    file_name="embeddings.npy",
                    mime="application/octet-stream",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ **Error exporting embeddings:** {str(e)}")

# Helper function for file download
def download_file(url: str, filename: str):
    response = requests.get(f"{API_BASE_URL}{url}")
    return response.content

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-secondary); font-size: 0.9em; opacity: 0.7;">
    <p>📦 Chunking Optimizer • Responsive Dark Theme • Advanced Vector Search</p>
</div>
""", unsafe_allow_html=True)
