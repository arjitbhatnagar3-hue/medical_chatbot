"""
FastAPI server for the medical chatbot.

Serves BOTH the API and the static frontend from a single origin, so the whole
app deploys as one process:

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   -> reports whether the model + index loaded OK
    POST /ask      -> {"question": "..."} -> {"answer": "...", "sources": [...]}
    /              -> the chat UI (frontend/index.html)
    /docs          -> Swagger API docs
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag import _init_error, ask_question

app = FastAPI(title="Medical Chatbot API")

# CORS is only needed if you serve the frontend from a DIFFERENT origin
# (e.g. a separate dev server). When the frontend is served by this app
# (the default), requests are same-origin and CORS is unnecessary.
FRONTEND_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if FRONTEND_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=FRONTEND_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def health():
    if _init_error is not None:
        return {"status": "error", "detail": str(_init_error)}
    return {"status": "ok"}


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/ask", response_model=Answer)
async def ask(question: Question):
    if not question.question or not question.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    # Offload the blocking LLM/embedding work to a threadpool.
    try:
        return await run_in_threadpool(ask_question, question.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- Serve the frontend from the same app (single-origin deploy) ---
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
