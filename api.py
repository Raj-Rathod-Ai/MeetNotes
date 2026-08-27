import os
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from utils.audio_processor import process_input, cleanup_files
from utils.exporter import export_as_pdf, export_as_markdown
from core.transcriber import transcribe_all, fetch_youtube_captions
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv(override=True)

app = FastAPI(
    title="MeetNote API",
    description="High-Performance Meeting & Video Intelligence API with Automatic Storage Cleanup",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UrlRequest(BaseModel):
    url: str
    language: Optional[str] = "english"
    model: Optional[str] = "base"


class ChatRequest(BaseModel):
    transcript: str
    question: str


class ExportRequest(BaseModel):
    title: str
    summary: str
    action_items: str
    key_decisions: str
    open_questions: str
    transcript: str


def run_pipeline_with_cleanup(source_path: str, language: str = "english", model_name: str = "base", temp_dir: Optional[str] = None) -> dict:
    """
    Executes the transcription and analysis pipeline, ensuring all temporary
    audio and chunk files are deleted immediately after transcription.
    """
    created_chunks = []
    transcript = None
    clean_source = source_path.strip()

    # Step 1: Check instant YouTube captions if URL
    if "youtube.com" in clean_source or "youtu.be" in clean_source:
        transcript = fetch_youtube_captions(clean_source)

    # Step 2: Fallback to audio processing if captions not available or if local file
    if not transcript:
        try:
            created_chunks = process_input(clean_source, work_dir=temp_dir)
            transcript = transcribe_all(created_chunks, language=language.lower(), model_name=model_name, source_url=clean_source)
            cleanup_files(created_chunks)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            cleanup_files(created_chunks)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e

    if not transcript:
        raise ValueError("Could not extract or transcribe audio from the provided source.")

    # Step 3: LLM Synthesis & Extraction (In-memory, zero disk footprint)
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "word_count": len(transcript.split()),
    }


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "MeetNote API",
        "endpoints": {
            "health": "/health",
            "process_url": "POST /api/process-url",
            "upload_file": "POST /api/upload-file",
            "chat": "POST /api/chat",
            "export_pdf": "POST /api/export-pdf",
            "export_md": "POST /api/export-md",
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "meetnote-api"}


@app.post("/api/process-url")
def process_url(req: UrlRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="A valid YouTube URL is required.")

    session_dir = tempfile.mkdtemp(prefix="meetnote_url_")
    try:
        result = run_pipeline_with_cleanup(
            source_path=req.url.strip(),
            language=req.language or "english",
            model_name=req.model or "base",
            temp_dir=session_dir,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    language: str = Form("english"),
    model: str = Form("base"),
):
    session_dir = tempfile.mkdtemp(prefix="meetnote_upload_")
    local_file_path = os.path.join(session_dir, file.filename)

    try:
        with open(local_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = run_pipeline_with_cleanup(
            source_path=local_file_path,
            language=language,
            model_name=model,
            temp_dir=session_dir,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/chat")
def chat_with_meeting(req: ChatRequest):
    if not req.transcript.strip() or not req.question.strip():
        raise HTTPException(status_code=400, detail="Both transcript and question are required.")

    try:
        chain = build_rag_chain(req.transcript)
        answer = ask_question(chain, req.question)
        return {"question": req.question, "answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/export-pdf")
def export_pdf_endpoint(req: ExportRequest):
    try:
        pdf_bytes = export_as_pdf(req.model_dump())
        filename = f"{req.title.replace(' ', '_')}_notes.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/export-md")
def export_md_endpoint(req: ExportRequest):
    try:
        md_content = export_as_markdown(req.model_dump())
        filename = f"{req.title.replace(' ', '_')}_notes.md"
        return Response(
            content=md_content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
