"""
FastAPI server for the medical chatbot.

Run from the PROJECT ROOT with:
    uvicorn app.main:app --reload

Endpoints:
    GET  /         -> {"status": "Chatbot is working"}
    GET  /health   -> reports whether the model + index loaded OK
    POST /ask      -> {"question": "..."} -> {"answer": "...", "sources": [...]}
"""
import os

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag import _init_error, ask_question

app = FastAPI(title="Medical Chatbot API")

# Restrict CORS to your real frontend origin(s). Wildcard + credentials is invalid.
FRONTEND_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "Chatbot is working"}


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

    # Offload the blocking LLM/embedding work to a threadpool so the event loop stays free.
    try:
        return await run_in_threadpool(ask_question, question.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
