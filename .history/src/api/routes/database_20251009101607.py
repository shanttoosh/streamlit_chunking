# Database API Routes
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from ...core.database import (
    test_database_connection, list_database_tables, import_database_table,
    connect_mysql, connect_postgresql, get_table_list, import_table_to_dataframe,
    is_large_table, import_large_table_to_dataframe
)

router = APIRouter()

@router.post("/db/test_connection")
async def db_test_connection(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    """Test database connection"""
    try:
        result = test_database_connection(db_type, host, port, username, password, database)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/db/list_tables")
async def db_list_tables(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    """List tables in database"""
    try:
        result = list_database_tables(db_type, host, port, username, password, database)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/db/import_one")
async def db_import_one(
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
        result = import_database_table(db_type, host, port, username, password, database, table_name)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}