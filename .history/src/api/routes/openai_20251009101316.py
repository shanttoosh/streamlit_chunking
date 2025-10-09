# OpenAI-Compatible API Routes
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
import json
from ...core.embedding import OpenAIEmbeddingAPI

router = APIRouter()

@router.post("/v1/embeddings")
async def openai_embeddings(
    model: str = Form("text-embedding-ada-002"),
    input: str = Form(...),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None)
):
    """OpenAI-compatible embeddings endpoint"""
    try:
        embedding_api = OpenAIEmbeddingAPI(
            model_name=model,
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        
        # Handle both string and list of strings
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input
            
        embeddings = embedding_api.encode(texts)
        
        # Format response in OpenAI standard
        response_data = {
            "object": "list",
            "data": [],
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        }
        
        for i, embedding in enumerate(embeddings):
            response_data["data"].append({
                "object": "embedding",
                "embedding": embedding.tolist(),
                "index": i
            })
            
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

@router.post("/v1/chat/completions")
async def openai_chat_completions(
    model: str = Form("gpt-3.5-turbo"),
    messages: str = Form(...),
    max_tokens: Optional[int] = Form(1000),
    temperature: Optional[float] = Form(0.7),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None)
):
    """OpenAI-compatible chat completions endpoint (requires external OpenAI API)"""
    try:
        import openai
        
        if openai_api_key:
            openai.api_key = openai_api_key
        if openai_base_url:
            openai.base_url = openai_base_url
            
        # Parse messages from JSON string
        messages_list = json.loads(messages)
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages_list,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return JSONResponse(content=response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion error: {str(e)}")
