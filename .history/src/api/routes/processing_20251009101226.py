# Processing API Routes
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import tempfile
import os
from typing import Optional
import json
from ...core.pipelines import run_fast_pipeline, run_config1_pipeline, run_deep_config_pipeline, process_large_file, process_file_direct
from ...core.database import test_database_connection, list_database_tables, import_database_table
from ...core.preprocessing import preprocess_csv_enhanced, convert_column_types, profile_nulls_enhanced, apply_null_strategies_enhanced, analyze_duplicates_enhanced, remove_duplicates_enhanced, remove_stopwords_from_text_column_enhanced, process_text_enhanced
from ...core.chunking import chunk_fixed_enhanced, chunk_semantic_cluster_enhanced, document_based_chunking_enhanced, chunk_recursive_keyvalue_enhanced
from ...core.embedding import embed_texts_enhanced
from ...core.storage import store_faiss_enhanced, store_chroma_enhanced

router = APIRouter()

@router.post("/run_fast")
async def run_fast(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(False),
    batch_size: int = Form(256)
):
    """Fast mode processing endpoint"""
    try:
        df = None
        
        # Handle file upload
        if file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Check if file is large
                file_size = os.path.getsize(tmp_file_path)
                if process_large_files and file_size > 10 * 1024 * 1024:  # 10MB
                    result = process_large_file(tmp_file_path, "fast", 
                                              use_openai=use_openai,
                                              openai_api_key=openai_api_key,
                                              openai_base_url=openai_base_url,
                                              use_turbo=use_turbo,
                                              batch_size=batch_size)
                else:
                    df = pd.read_csv(tmp_file_path)
                    result = run_fast_pipeline(df, 
                                            use_openai=use_openai,
                                            openai_api_key=openai_api_key,
                                            openai_base_url=openai_base_url,
                                            use_turbo=use_turbo,
                                            batch_size=batch_size)
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        # Handle database import
        elif db_type and host and port and username and password and database and table_name:
            db_result = import_database_table(db_type, host, port, username, password, database, table_name)
            if not db_result["success"]:
                return {"error": db_result["error"]}
            
            df = db_result["dataframe"]
            result = run_fast_pipeline(df,
                                    db_type=db_type,
                                    db_config={"host": host, "port": port, "username": username, "password": password, "database": database},
                                    use_openai=use_openai,
                                    openai_api_key=openai_api_key,
                                    openai_base_url=openai_base_url,
                                    use_turbo=use_turbo,
                                    batch_size=batch_size)
        
        else:
            return {"error": "Either file or database parameters must be provided"}
        
        return {"success": True, "result": result}
        
    except Exception as e:
        return {"error": str(e)}

@router.post("/run_config1")
async def run_config1(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    document_key_column: str = Form(None),
    token_limit: int = Form(2000),
    retrieval_metric: str = Form("cosine"),
    model_choice: str = Form("paraphrase-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss"),
    apply_default_preprocessing: bool = Form(True),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(False),
    batch_size: int = Form(256)
):
    """Config-1 mode processing endpoint"""
    try:
        df = None
        
        # Handle file upload
        if file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Check if file is large
                file_size = os.path.getsize(tmp_file_path)
                if process_large_files and file_size > 10 * 1024 * 1024:  # 10MB
                    result = process_large_file(tmp_file_path, "config1",
                                              chunk_method=chunk_method,
                                              chunk_size=chunk_size,
                                              overlap=overlap,
                                              document_key_column=document_key_column,
                                              token_limit=token_limit,
                                              retrieval_metric=retrieval_metric,
                                              model_choice=model_choice,
                                              storage_choice=storage_choice,
                                              apply_default_preprocessing=apply_default_preprocessing,
                                              use_openai=use_openai,
                                              openai_api_key=openai_api_key,
                                              openai_base_url=openai_base_url,
                                              use_turbo=use_turbo,
                                              batch_size=batch_size)
                else:
                    df = pd.read_csv(tmp_file_path)
                    result = run_config1_pipeline(df,
                                                chunk_method=chunk_method,
                                                chunk_size=chunk_size,
                                                overlap=overlap,
                                                document_key_column=document_key_column,
                                                token_limit=token_limit,
                                                retrieval_metric=retrieval_metric,
                                                model_choice=model_choice,
                                                storage_choice=storage_choice,
                                                apply_default_preprocessing=apply_default_preprocessing,
                                                use_openai=use_openai,
                                                openai_api_key=openai_api_key,
                                                openai_base_url=openai_base_url,
                                                use_turbo=use_turbo,
                                                batch_size=batch_size)
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        # Handle database import
        elif db_type and host and port and username and password and database and table_name:
            db_result = import_database_table(db_type, host, port, username, password, database, table_name)
            if not db_result["success"]:
                return {"error": db_result["error"]}
            
            df = db_result["dataframe"]
            result = run_config1_pipeline(df,
                                        chunk_method=chunk_method,
                                        chunk_size=chunk_size,
                                        overlap=overlap,
                                        document_key_column=document_key_column,
                                        token_limit=token_limit,
                                        retrieval_metric=retrieval_metric,
                                        model_choice=model_choice,
                                        storage_choice=storage_choice,
                                        apply_default_preprocessing=apply_default_preprocessing,
                                        use_openai=use_openai,
                                        openai_api_key=openai_api_key,
                                        openai_base_url=openai_base_url,
                                        use_turbo=use_turbo,
                                        batch_size=batch_size)
        
        else:
            return {"error": "Either file or database parameters must be provided"}
        
        return {"success": True, "result": result}
        
    except Exception as e:
        return {"error": str(e)}

@router.post("/run_deep_config")
async def run_deep_config(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None),
    preprocessing_config: str = Form("{}"),
    chunking_config: str = Form("{}"),
    embedding_config: str = Form("{}"),
    storage_config: str = Form("{}")
):
    """Deep config mode processing endpoint"""
    try:
        df = None
        
        # Parse configuration
        config_dict = {
            'preprocessing': json.loads(preprocessing_config),
            'chunking': json.loads(chunking_config),
            'embedding': json.loads(embedding_config),
            'storage': json.loads(storage_config)
        }
        
        # Handle file upload
        if file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                df = pd.read_csv(tmp_file_path)
                result = run_deep_config_pipeline(df, config_dict)
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        # Handle database import
        elif db_type and host and port and username and password and database and table_name:
            db_result = import_database_table(db_type, host, port, username, password, database, table_name)
            if not db_result["success"]:
                return {"error": db_result["error"]}
            
            df = db_result["dataframe"]
            result = run_deep_config_pipeline(df, config_dict)
        
        else:
            return {"error": "Either file or database parameters must be provided"}
        
        return {"success": True, "result": result}
        
    except Exception as e:
        return {"error": str(e)}

# Deep Config Step-by-Step Endpoints
@router.post("/deep_config/preprocess")
async def deep_config_preprocess(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None)
):
    """Step 1: Load and preprocess data"""
    try:
        df = None
        
        # Handle file upload
        if file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                df = pd.read_csv(tmp_file_path)
            finally:
                os.unlink(tmp_file_path)
        
        # Handle database import
        elif db_type and host and port and username and password and database and table_name:
            db_result = import_database_table(db_type, host, port, username, password, database, table_name)
            if not db_result["success"]:
                return {"error": db_result["error"]}
            df = db_result["dataframe"]
        
        else:
            return {"error": "Either file or database parameters must be provided"}
        
        # Apply basic preprocessing
        df_processed = preprocess_csv_enhanced(df)
        
        return {
            "success": True,
            "original_rows": len(df),
            "processed_rows": len(df_processed),
            "columns": list(df_processed.columns),
            "data_types": df_processed.dtypes.to_dict(),
            "preview": df_processed.head().to_dict('records')
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.post("/deep_config/type_convert")
async def deep_config_type_convert(
    type_conversions: str = Form(...)
):
    """Step 2: Convert data types"""
    try:
        conversions = json.loads(type_conversions)
        # This would typically work with session state or stored data
        # For now, return success
        return {"success": True, "conversions": conversions}
    except Exception as e:
        return {"error": str(e)}

@router.post("/deep_config/null_handle")
async def deep_config_null_handle(
    null_strategies: str = Form(...)
):
    """Step 3: Handle null values"""
    try:
        strategies = json.loads(null_strategies)
        # This would typically work with session state or stored data
        return {"success": True, "strategies": strategies}
    except Exception as e:
        return {"error": str(e)}

@router.post("/deep_config/duplicates")
async def deep_config_duplicates(
    strategy: str = Form("keep_first")
):
    """Step 4: Handle duplicates"""
    try:
        # This would typically work with session state or stored data
        return {"success": True, "strategy": strategy}
    except Exception as e:
        return {"error": str(e)}

@router.post("/deep_config/stopwords")
async def deep_config_stopwords(
    remove_stopwords: bool = Form(False)
):
    """Step 5: Remove stop words"""
    try:
        # This would typically work with session state or stored data
        return {"success": True, "remove_stopwords": remove_stopwords}
    except Exception as e:
        return {"error": str(e)}

@router.post("/deep_config/normalize")
async def deep_config_normalize(
    text_processing: str = Form("none")
):
    """Step 6: Text normalization"""
    try:
        # This would typically work with session state or stored data
        return {"success": True, "text_processing": text_processing}
    except Exception as e:
        return {"error": str(e)}

@router.post("/deep_config/chunk")
async def deep_config_chunk(
    chunk_params: str = Form(...)
):
    """Step 7: Chunk data"""
    try:
        params = json.loads(chunk_params)
        # This would typically work with session state or stored data
        return {"success": True, "chunk_params": params}
    except Exception as e:
        return {"error": str(e)}

@router.post("/deep_config/embed")
async def deep_config_embed(
    embed_params: str = Form(...)
):
    """Step 8: Generate embeddings"""
    try:
        params = json.loads(embed_params)
        # This would typically work with session state or stored data
        return {"success": True, "embed_params": params}
    except Exception as e:
        return {"error": str(e)}

@router.post("/deep_config/store")
async def deep_config_store(
    store_params: str = Form(...)
):
    """Step 9: Store embeddings"""
    try:
        params = json.loads(store_params)
        # This would typically work with session state or stored data
        return {"success": True, "store_params": params}
    except Exception as e:
        return {"error": str(e)}

@router.post("/upload_large_file")
async def upload_large_file(
    file: UploadFile = File(...),
    processing_mode: str = Form("fast"),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    use_turbo: bool = Form(True),
    batch_size: int = Form(256)
):
    """Upload and process large files"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process large file
            result = process_large_file(tmp_file_path, processing_mode,
                                      use_openai=use_openai,
                                      openai_api_key=openai_api_key,
                                      openai_base_url=openai_base_url,
                                      use_turbo=use_turbo,
                                      batch_size=batch_size)
            
            return {"success": True, "result": result}
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
    except Exception as e:
        return {"error": str(e)}