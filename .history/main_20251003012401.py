# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import pandas as pd
import io
import numpy as np
import tempfile
import os
from backend import (
    run_fast_pipeline, 
    run_config1_pipeline, 
    run_deep_pipeline, 
    retrieve_similar,
    export_chunks,
    export_embeddings,
    get_system_info,
    get_file_info,
    connect_mysql,
    connect_postgresql,
    get_table_list,
    import_table_to_dataframe
)

app = FastAPI(title="Chunking Optimizer API", version="1.0")
# ---------------------------
# DB IMPORT (single-table test)
# ---------------------------
@app.post("/db/test_connection")
async def db_test_connection(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"status": "error", "message": "Unsupported db_type"}
        conn.close()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/db/list_tables")
async def db_list_tables(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        tables = get_table_list(conn, db_type)
        conn.close()
        return {"tables": tables}
    except Exception as e:
        return {"error": str(e)}

@app.post("/db/import_one")
async def db_import_one(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...),
    table_name: str = Form(...)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        df = import_table_to_dataframe(conn, table_name)
        conn.close()
        
        # Create enhanced file info for database source
        from backend import create_db_data_source_info
        import datetime
        data_source_info = create_db_data_source_info(
            db_type, host, port, database, table_name, 
            datetime.datetime.now().isoformat()
        )
        
        result = run_fast_pipeline(df, file_info=data_source_info)
        return {"mode": "fast", "summary": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/db/run_config1")
async def db_run_config1(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...),
    table_name: str = Form(...),
    null_handling: str = Form("keep"),
    fill_value: str = Form("Unknown"),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    model_choice: str = Form("all-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss")
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        
        df = import_table_to_dataframe(conn, table_name)
        conn.close()
        
        # Create enhanced file info for database source
        from backend import create_db_data_source_info
        import datetime
        data_source_info = create_db_data_source_info(
            db_type, host, port, database, table_name, 
            datetime.datetime.now().isoformat()
        )
        
        result = run_config1_pipeline(df, null_handling, fill_value, chunk_method,
                                    chunk_size, overlap, model_choice, storage_choice, file_info=data_source_info)
        return {"mode": "config1", "summary": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/db/run_deep")
async def db_run_deep(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...),
    table_name: str = Form(...),
    null_handling: str = Form("keep"),
    fill_value: str = Form("Unknown"),
    remove_stopwords: bool = Form(False),
    lowercase: bool = Form(True),
    stemming: bool = Form(False),
    lemmatization: bool = Form(False),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    model_choice: str = Form("all-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss")
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        
        df = import_table_to_dataframe(conn, table_name)
        conn.close()
        
        # Create enhanced file info for database source
        from backend import create_db_data_source_info
        import datetime
        data_source_info = create_db_data_source_info(
            db_type, host, port, database, table_name, 
            datetime.datetime.now().isoformat()
        )
        
        result = run_deep_pipeline(df, null_handling, fill_value, remove_stopwords,
                                 lowercase, stemming, lemmatization, chunk_method,
                                 chunk_size, overlap, model_choice, storage_choice, file_info=data_source_info)
        return {"mode": "deep", "summary": result}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# FAST MODE
# ---------------------------
@app.post("/run_fast")
async def run_fast(file: UploadFile = File(...), db_type: str = Form("sqlite")):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    file_info = {
        "filename": file.filename,
        "file_size": len(contents),
        "upload_time": pd.Timestamp.now().isoformat()
    }
    
    result = run_fast_pipeline(df, db_type, file_info=file_info)
    return {"mode": "fast", "summary": result}

# ---------------------------
# CONFIG-1 MODE
# ---------------------------
@app.post("/run_config1")
async def run_config1(
    file: UploadFile = File(...),
    null_handling: str = Form("keep"),
    fill_value: str = Form("Unknown"),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    model_choice: str = Form("all-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss")
):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    file_info = {
        "filename": file.filename,
        "file_size": len(contents),
        "upload_time": pd.Timestamp.now().isoformat()
    }
    
    result = run_config1_pipeline(df, null_handling, fill_value, chunk_method,
                                  chunk_size, overlap, model_choice, storage_choice, file_info=file_info)
    return {"mode": "config1", "summary": result}

# ---------------------------
# DEEP CONFIG MODE
# ---------------------------
@app.post("/run_deep")
async def run_deep(
    file: UploadFile = File(...),
    null_handling: str = Form("keep"),
    fill_value: str = Form("Unknown"),
    remove_stopwords: bool = Form(False),
    lowercase: bool = Form(True),
    stemming: bool = Form(False),
    lemmatization: bool = Form(False),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    model_choice: str = Form("all-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss")
):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    file_info = {
        "filename": file.filename,
        "file_size": len(contents),
        "upload_time": pd.Timestamp.now().isoformat()
    }
    
    result = run_deep_pipeline(df, null_handling, fill_value, remove_stopwords,
                               lowercase, stemming, lemmatization, chunk_method,
                               chunk_size, overlap, model_choice, storage_choice, file_info=file_info)
    return {"mode": "deep", "summary": result}

# ---------------------------
# RETRIEVAL ENDPOINTS
# ---------------------------
@app.post("/retrieve")
async def retrieve(query: str = Form(...), k: int = Form(5)):
    """Retrieve similar chunks after running any pipeline"""
    result = retrieve_similar(query, k)
    return result

    

# ---------------------------
# SYSTEM INFO ENDPOINTS
# ---------------------------
@app.get("/system_info")
async def system_info():
    """Get system information"""
    return get_system_info()

@app.get("/file_info")
async def file_info():
    """Get file information"""
    return get_file_info()

# ---------------------------
# EXPORT ENDPOINTS
# ---------------------------
@app.get("/export/chunks")
async def export_chunks_file():
    """Export chunks as text file"""
    chunks_text = export_chunks()
    if not chunks_text:
        return {"error": "No chunks available"}
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(chunks_text)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="chunks.txt", media_type="text/plain")

@app.get("/export/embeddings")
async def export_embeddings_file():
    """Export embeddings as numpy file"""
    embeddings = export_embeddings()
    if embeddings is None:
        return {"error": "No embeddings available"}
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, embeddings)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="embeddings.npy", media_type="application/octet-stream")

# ---------------------------
# HEALTH CHECK
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Chunking Optimizer API is running", "version": "1.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}