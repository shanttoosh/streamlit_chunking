# Export API Routes
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
from ...core.retrieval import export_chunks, export_embeddings, export_embeddings_json, export_embeddings_text

router = APIRouter()

@router.get("/export/chunks")
async def export_chunks_endpoint():
    """Export chunks as CSV"""
    try:
        chunks_csv = export_chunks()
        if not chunks_csv:
            return {"error": "No chunks available for export"}
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(chunks_csv)
            tmp_file_path = tmp_file.name
        
        return FileResponse(
            tmp_file_path,
            media_type='text/csv',
            filename='chunks.csv',
            background=lambda: os.unlink(tmp_file_path)  # Clean up after download
        )
    except Exception as e:
        return {"error": str(e)}

@router.get("/export/embeddings")
async def export_embeddings_endpoint():
    """Export embeddings as numpy array"""
    try:
        embeddings = export_embeddings()
        if embeddings is None:
            return {"error": "No embeddings available for export"}
        
        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        return {"embeddings": embeddings_list}
    except Exception as e:
        return {"error": str(e)}

@router.get("/export/embeddings_text")
async def export_embeddings_text_endpoint():
    """Export embeddings as text format"""
    try:
        embeddings_text = export_embeddings_text()
        if not embeddings_text:
            return {"error": "No embeddings available for export"}
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(embeddings_text)
            tmp_file_path = tmp_file.name
        
        return FileResponse(
            tmp_file_path,
            media_type='text/plain',
            filename='embeddings.txt',
            background=lambda: os.unlink(tmp_file_path)  # Clean up after download
        )
    except Exception as e:
        return {"error": str(e)}

@router.get("/export/embeddings_json")
async def export_embeddings_json_endpoint():
    """Export embeddings as JSON"""
    try:
        embeddings_json = export_embeddings_json()
        if not embeddings_json:
            return {"error": "No embeddings available for export"}
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_file.write(embeddings_json)
            tmp_file_path = tmp_file.name
        
        return FileResponse(
            tmp_file_path,
            media_type='application/json',
            filename='embeddings.json',
            background=lambda: os.unlink(tmp_file_path)  # Clean up after download
        )
    except Exception as e:
        return {"error": str(e)}

# Deep Config Export Endpoints
@router.get("/deep_config/export/preprocessed")
async def export_deep_config_preprocessed():
    """Export preprocessed data from deep config"""
    try:
        # This would typically work with session state or stored data
        # For now, return a placeholder
        return {"error": "No preprocessed data available"}
    except Exception as e:
        return {"error": str(e)}

@router.get("/deep_config/export/chunks")
async def export_deep_config_chunks():
    """Export chunks from deep config"""
    try:
        chunks_csv = export_chunks()
        if not chunks_csv:
            return {"error": "No chunks available for export"}
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(chunks_csv)
            tmp_file_path = tmp_file.name
        
        return FileResponse(
            tmp_file_path,
            media_type='text/csv',
            filename='deep_config_chunks.csv',
            background=lambda: os.unlink(tmp_file_path)  # Clean up after download
        )
    except Exception as e:
        return {"error": str(e)}

@router.get("/deep_config/export/embeddings")
async def export_deep_config_embeddings():
    """Export embeddings from deep config"""
    try:
        embeddings_json = export_embeddings_json()
        if not embeddings_json:
            return {"error": "No embeddings available for export"}
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_file.write(embeddings_json)
            tmp_file_path = tmp_file.name
        
        return FileResponse(
            tmp_file_path,
            media_type='application/json',
            filename='deep_config_embeddings.json',
            background=lambda: os.unlink(tmp_file_path)  # Clean up after download
        )
    except Exception as e:
        return {"error": str(e)}