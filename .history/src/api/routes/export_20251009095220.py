# Export Routes
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import json
import tempfile
import os
from typing import Optional
from ...core.pipelines import current_chunks, current_embeddings, current_df, load_state

router = APIRouter()

@router.get("/export/chunks")
async def export_chunks():
    """Export current chunks as CSV"""
    try:
        # Load state if globals are empty
        if current_chunks is None:
            load_state()
        
        if not current_chunks:
            return JSONResponse(status_code=404, content={"error": "No chunks available for export"})
        
        # Create DataFrame with all chunks
        df = pd.DataFrame({
            'chunk_id': range(1, len(current_chunks) + 1),
            'chunk_text': current_chunks
        })
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return FileResponse(
            path=temp_file.name,
            filename="chunks.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/export/embeddings")
async def export_embeddings():
    """Export current embeddings as numpy array"""
    try:
        # Load state if globals are empty
        if current_embeddings is None:
            load_state()
        
        if current_embeddings is None:
            return JSONResponse(status_code=404, content={"error": "No embeddings available for export"})
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        import numpy as np
        np.save(temp_file.name, current_embeddings)
        temp_file.close()
        
        return FileResponse(
            path=temp_file.name,
            filename="embeddings.npy",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/export/embeddings_text")
async def export_embeddings_text():
    """Export current embeddings as JSON format"""
    try:
        # Load state if globals are empty
        if current_embeddings is None or current_chunks is None:
            load_state()
        
        if current_embeddings is None or current_chunks is None:
            return JSONResponse(status_code=404, content={"error": "No embeddings or chunks available for export"})
        
        # Create JSON structure with all embeddings
        embeddings_data = {
            "total_chunks": len(current_chunks),
            "vector_dimension": current_embeddings.shape[1] if len(current_embeddings.shape) > 1 else 0,
            "embeddings": []
        }
        
        for i, (chunk, embedding) in enumerate(zip(current_chunks, current_embeddings)):
            embeddings_data["embeddings"].append({
                "chunk_id": i + 1,
                "chunk_text": chunk,
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            })
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(embeddings_data, temp_file, indent=2)
        temp_file.close()
        
        return FileResponse(
            path=temp_file.name,
            filename="embeddings.json",
            media_type="application/json"
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/export/preprocessed")
async def export_preprocessed():
    """Export preprocessed data as CSV"""
    try:
        # Load state if globals are empty
        if current_df is None:
            load_state()
        
        if current_df is None:
            return JSONResponse(status_code=404, content={"error": "No preprocessed data available for export"})
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        current_df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return FileResponse(
            path=temp_file.name,
            filename="preprocessed_data.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/export/deep_chunks")
async def export_deep_chunks():
    """Export deep config chunks with metadata"""
    try:
        # Load state if globals are empty
        if current_chunks is None:
            load_state()
        
        if not current_chunks:
            return JSONResponse(status_code=404, content={"error": "No chunks available for export"})
        
        # Create DataFrame with chunks and metadata
        df = pd.DataFrame({
            'chunk_id': range(1, len(current_chunks) + 1),
            'chunk_text': current_chunks
        })
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return FileResponse(
            path=temp_file.name,
            filename="deep_chunks.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/export/deep_embeddings")
async def export_deep_embeddings():
    """Export deep config embeddings with metadata"""
    try:
        # Load state if globals are empty
        if current_embeddings is None or current_chunks is None:
            load_state()
        
        if current_embeddings is None or current_chunks is None:
            return JSONResponse(status_code=404, content={"error": "No embeddings or chunks available for export"})
        
        # Create enhanced JSON structure
        embeddings_data = {
            "total_chunks": len(current_chunks),
            "vector_dimension": current_embeddings.shape[1] if len(current_embeddings.shape) > 1 else 0,
            "export_type": "deep_config",
            "embeddings": []
        }
        
        for i, (chunk, embedding) in enumerate(zip(current_chunks, current_embeddings)):
            embeddings_data["embeddings"].append({
                "chunk_id": i + 1,
                "chunk_text": chunk,
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                "metadata": {
                    "chunking_method": "deep_config",
                    "chunk_index": i
                }
            })
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(embeddings_data, temp_file, indent=2)
        temp_file.close()
        
        return FileResponse(
            path=temp_file.name,
            filename="deep_embeddings.json",
            media_type="application/json"
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/export/status")
async def get_export_status():
    """Get current export status"""
    try:
        # Load state if globals are empty
        if current_chunks is None:
            load_state()
        
        status = {
            "chunks_available": current_chunks is not None,
            "embeddings_available": current_embeddings is not None,
            "preprocessed_data_available": current_df is not None,
            "total_chunks": len(current_chunks) if current_chunks else 0,
            "total_embeddings": len(current_embeddings) if current_embeddings is not None else 0
        }
        
        return status
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
