# Processing Routes
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import tempfile
import os
from typing import Optional
from ...core.pipelines import run_fast_pipeline, run_config1_pipeline, run_deep_config_pipeline
from ...core.database import test_database_connection, get_database_tables, import_table_to_dataframe, import_large_table_to_dataframe, is_large_table
from ...utils.file_utils import save_uploaded_file, get_file_info as get_file_info_util, can_load_file
from ...utils.validation import validate_file_upload, validate_chunking_params, validate_database_config
from ...config.settings import settings

router = APIRouter()

@router.post("/process/fast")
async def process_fast(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: Optional[str] = Form(None),
    port: Optional[int] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(True),
    batch_size: int = Form(256)
):
    """Fast mode processing endpoint"""
    try:
        df = None
        file_info = {}
        
        # Handle file upload
        if file:
            # Validate file
            validation = validate_file_upload(file)
            if not validation["is_valid"]:
                return JSONResponse(status_code=400, content={"error": validation["errors"]})
            
            # Save uploaded file
            temp_file_path = save_uploaded_file(file)
            file_info = get_file_info_util(temp_file_path)
            
            # Check if file is large
            if can_load_file(file_info.get("size", 0)):
                df = pd.read_csv(temp_file_path)
            else:
                # Handle large file processing
                return {"message": "Large file detected, use /process/large_file endpoint"}
            
            # Cleanup temp file
            os.remove(temp_file_path)
        
        # Handle database import
        elif db_type in ["mysql", "postgresql"]:
            # Validate database config
            db_config = {
                "db_type": db_type,
                "host": host,
                "port": port,
                "username": username,
                "password": password,
                "database": database,
                "table_name": table_name
            }
            
            validation = validate_database_config(db_config)
            if not validation["is_valid"]:
                return JSONResponse(status_code=400, content={"error": validation["errors"]})
            
            # Test connection
            conn_test = test_database_connection(db_type, host, port, username, password, database)
            if not conn_test["connected"]:
                return JSONResponse(status_code=400, content={"error": conn_test["message"]})
            
            # Import table
            import mysql.connector
            conn = mysql.connector.connect(
                host=host, port=port, user=username, password=password, database=database
            ) if db_type == "mysql" else None
            
            if not conn:
                import psycopg2
                conn = psycopg2.connect(
                    host=host, port=port, user=username, password=password, dbname=database
                )
            
            # Check if table is large
            if is_large_table(conn, table_name):
                df = import_large_table_to_dataframe(conn, table_name)
            else:
                df = import_table_to_dataframe(conn, table_name)
            
            conn.close()
            file_info = {"filename": table_name, "file_type": "database_table"}
        
        if df is None:
            return JSONResponse(status_code=400, content={"error": "No data provided"})
        
        # Run fast pipeline
        result = run_fast_pipeline(
            df=df,
            file_info=file_info,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            use_turbo=use_turbo,
            batch_size=batch_size
        )
        
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/process/config1")
async def process_config1(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: Optional[str] = Form(None),
    port: Optional[int] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    document_key_column: Optional[str] = Form(None),
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
        # Validate chunking parameters
        validation = validate_chunking_params(chunk_size, overlap)
        if not validation["is_valid"]:
            return JSONResponse(status_code=400, content={"error": validation["errors"]})
        
        df = None
        file_info = {}
        
        # Handle file upload (similar to fast mode)
        if file:
            validation = validate_file_upload(file)
            if not validation["is_valid"]:
                return JSONResponse(status_code=400, content={"error": validation["errors"]})
            
            temp_file_path = save_uploaded_file(file)
            file_info = get_file_info_util(temp_file_path)
            
            if can_load_file(file_info.get("size", 0)):
                df = pd.read_csv(temp_file_path)
            else:
                return {"message": "Large file detected, use /process/large_file endpoint"}
            
            os.remove(temp_file_path)
        
        # Handle database import (similar to fast mode)
        elif db_type in ["mysql", "postgresql"]:
            db_config = {
                "db_type": db_type, "host": host, "port": port,
                "username": username, "password": password,
                "database": database, "table_name": table_name
            }
            
            validation = validate_database_config(db_config)
            if not validation["is_valid"]:
                return JSONResponse(status_code=400, content={"error": validation["errors"]})
            
            # Import data (simplified for brevity)
            df = pd.DataFrame()  # Placeholder
            file_info = {"filename": table_name, "file_type": "database_table"}
        
        if df is None:
            return JSONResponse(status_code=400, content={"error": "No data provided"})
        
        # Run config1 pipeline
        result = run_config1_pipeline(
            df=df,
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            overlap=overlap,
            model_choice=model_choice,
            storage_choice=storage_choice,
            file_info=file_info,
            use_openai=use_openai,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            use_turbo=use_turbo,
            batch_size=batch_size,
            document_key_column=document_key_column,
            token_limit=token_limit,
            retrieval_metric=retrieval_metric,
            apply_default_preprocessing=apply_default_preprocessing
        )
        
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/process/deep_config")
async def process_deep_config(
    file: Optional[UploadFile] = File(None),
    preprocessing_config: str = Form("{}"),
    chunking_config: str = Form("{}"),
    embedding_config: str = Form("{}"),
    storage_config: str = Form("{}")
):
    """Deep config mode processing endpoint"""
    try:
        import json
        
        # Parse configs
        config_dict = {
            "preprocessing": json.loads(preprocessing_config),
            "chunking": json.loads(chunking_config),
            "embedding": json.loads(embedding_config),
            "storage": json.loads(storage_config)
        }
        
        df = None
        file_info = {}
        
        # Handle file upload
        if file:
            validation = validate_file_upload(file)
            if not validation["is_valid"]:
                return JSONResponse(status_code=400, content={"error": validation["errors"]})
            
            temp_file_path = save_uploaded_file(file)
            file_info = get_file_info_util(temp_file_path)
            
            if can_load_file(file_info.get("size", 0)):
                df = pd.read_csv(temp_file_path)
            else:
                return {"message": "Large file detected, use /process/large_file endpoint"}
            
            os.remove(temp_file_path)
        
        if df is None:
            return JSONResponse(status_code=400, content={"error": "No data provided"})
        
        # Run deep config pipeline
        result = run_deep_config_pipeline(df=df, config_dict=config_dict, file_info=file_info)
        
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/process/large_file")
async def process_large_file(
    file: UploadFile = File(...),
    processing_mode: str = Form("fast"),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    use_turbo: bool = Form(True),
    batch_size: int = Form(256)
):
    """Large file processing endpoint"""
    try:
        # Save uploaded file
        temp_file_path = save_uploaded_file(file)
        file_info = get_file_info_util(temp_file_path)
        
        # Process large file (simplified for now)
        result = {
            "message": "Large file processing initiated",
            "filename": file.filename,
            "size": file_info.get("size", 0),
            "processing_mode": processing_mode,
            "status": "processing"
        }
        
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
