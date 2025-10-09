# Retrieval API Routes
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import json
from ...core.retrieval import retrieve_similar, retrieve_with_metadata, get_system_info, get_file_info, get_capabilities

router = APIRouter()

@router.post("/retrieve")
async def retrieve(
    query: str = Form(...),
    k: int = Form(5)
):
    """Semantic search endpoint"""
    try:
        result = retrieve_similar(query, k)
        return result
    except Exception as e:
        return {"error": str(e)}

@router.post("/retrieve_with_metadata")
async def retrieve_with_metadata_endpoint(
    query: str = Form(...),
    k: int = Form(5),
    metadata_filter: str = Form("{}")
):
    """Semantic search with metadata filtering"""
    try:
        filter_dict = json.loads(metadata_filter) if metadata_filter else None
        result = retrieve_with_metadata(query, k, filter_dict)
        return result
    except Exception as e:
        return {"error": str(e)}

@router.post("/v1/retrieve")
async def v1_retrieve(
    query: str = Form(...),
    model: str = Form("all-MiniLM-L6-v2"),
    n_results: int = Form(5)
):
    """OpenAI-compatible retrieval endpoint"""
    try:
        result = retrieve_similar(query, n_results)
        return result
    except Exception as e:
        return {"error": str(e)}

@router.get("/system_info")
async def system_info():
    """Get system information"""
    try:
        result = get_system_info()
        return result
    except Exception as e:
        return {"error": str(e)}

@router.get("/file_info")
async def file_info():
    """Get file information"""
    try:
        result = get_file_info()
        return result
    except Exception as e:
        return {"error": str(e)}

@router.get("/capabilities")
async def capabilities():
    """Get system capabilities"""
    try:
        result = get_capabilities()
        return result
    except Exception as e:
        return {"error": str(e)}