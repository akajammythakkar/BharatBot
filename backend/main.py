from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from rag import query_rag
from config import ADMIN_KEY, SUPPORTED_LANGUAGES
from language import normalize

app = FastAPI(title="BharatBot API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    # Optional; "auto"/empty means detect. Falls back to English.
    input_language: Optional[str] = None
    output_language: Optional[str] = None
    # Deprecated alias for output_language, kept for older clients.
    language: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    language: str  # same as output_language
    input_language: str
    output_language: str
    output_language_source: str  # prompt | selected | input | default
    search_query: str  # English query sent to the vector store
    sources: list
    web_sources: list = []


@app.get("/health")
def health():
    return {"status": "ok", "bot": "BharatBot", "version": "2.1"}


@app.get("/languages")
def languages():
    return {"languages": SUPPORTED_LANGUAGES, "default": "en"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    output_language = req.output_language or req.language
    try:
        normalize(req.input_language, SUPPORTED_LANGUAGES)
        normalize(output_language, SUPPORTED_LANGUAGES)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        result = query_rag(req.message, req.input_language, output_language)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/ingest")
def trigger_ingest(x_admin_key: str = Header(None)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        from ingest import ingest
        ingest()
        return {"status": "ingestion complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
