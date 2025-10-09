# Retrieval Routes
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import json
from ...core.pipelines import current_model, current_store_info, current_chunks, current_embeddings, load_state
from ...core.retrieval import retrieve_similar

router = APIRouter()

@router.post("/retrieve")
async def retrieve_similar_chunks(
    query: str = Form(...),
    k: int = Form(5),
    metadata_filter: Optional[str] = Form(None)
):
    """Semantic search endpoint"""
    try:
        # Load state if globals are empty
        if current_model is None or current_store_info is None:
            load_state()
        
        # Parse metadata filter if provided
        filter_dict = None
        if metadata_filter:
            try:
                filter_dict = json.loads(metadata_filter)
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"error": "Invalid metadata_filter JSON"})
        
        # Perform retrieval
        result = retrieve_similar(
            query=query,
            k=k,
            current_model=current_model,
            current_store_info=current_store_info,
            current_chunks=current_chunks,
            current_embeddings=current_embeddings
        )
        
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/retrieve/advanced")
async def retrieve_advanced(
    query: str = Form(...),
    k: int = Form(5),
    metadata_filter: Optional[str] = Form(None),
    similarity_threshold: Optional[float] = Form(None)
):
    """Advanced semantic search with filtering"""
    try:
        # Load state if globals are empty
        if current_model is None or current_store_info is None:
            load_state()
        
        # Parse metadata filter if provided
        filter_dict = None
        if metadata_filter:
            try:
                filter_dict = json.loads(metadata_filter)
            except json.JSONDecodeError:
                return JSONResponse(status_code=400, content={"error": "Invalid metadata_filter JSON"})
        
        # Perform retrieval
        result = retrieve_similar(
            query=query,
            k=k,
            current_model=current_model,
            current_store_info=current_store_info,
            current_chunks=current_chunks,
            current_embeddings=current_embeddings
        )
        
        # Apply similarity threshold if provided
        if similarity_threshold and "results" in result:
            filtered_results = [
                r for r in result["results"] 
                if r.get("similarity", 0) >= similarity_threshold
            ]
            result["results"] = filtered_results
            result["filtered_count"] = len(filtered_results)
        
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/retrieve/status")
async def get_retrieval_status():
    """Get current retrieval system status"""
    try:
        # Load state if globals are empty
        if current_model is None or current_store_info is None:
            load_state()
        
        status = {
            "model_loaded": current_model is not None,
            "store_loaded": current_store_info is not None,
            "chunks_loaded": current_chunks is not None,
            "embeddings_loaded": current_embeddings is not None,
            "total_chunks": len(current_chunks) if current_chunks else 0,
            "store_type": current_store_info.get("type") if current_store_info else None
        }
        
        return status
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
