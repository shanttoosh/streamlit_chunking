# Database Routes
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from ...core.database import test_database_connection, get_database_tables, import_table_to_dataframe, import_large_table_to_dataframe, is_large_table
from ...utils.validation import validate_database_config

router = APIRouter()

@router.post("/db/test_connection")
async def test_db_connection(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    """Test database connection"""
    try:
        # Validate database config
        db_config = {
            "db_type": db_type,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": database
        }
        
        validation = validate_database_config(db_config)
        if not validation["is_valid"]:
            return JSONResponse(status_code=400, content={"error": validation["errors"]})
        
        # Test connection
        result = test_database_connection(db_type, host, port, username, password, database)
        
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/db/list_tables")
async def list_db_tables(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    """Get list of tables from database"""
    try:
        # Validate database config
        db_config = {
            "db_type": db_type,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": database
        }
        
        validation = validate_database_config(db_config)
        if not validation["is_valid"]:
            return JSONResponse(status_code=400, content={"error": validation["errors"]})
        
        # Get tables
        result = get_database_tables(db_type, host, port, username, password, database)
        
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/db/import_one")
async def import_db_table(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...),
    table_name: str = Form(...)
):
    """Import single table from database"""
    try:
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
        
        # Test connection first
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
            message = "Large table imported successfully"
        else:
            df = import_table_to_dataframe(conn, table_name)
            message = "Table imported successfully"
        
        conn.close()
        
        return {
            "success": True,
            "message": message,
            "table_name": table_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist()
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
