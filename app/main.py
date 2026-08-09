from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.rag import ask_question

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"Chatbot is working"}

class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(question: Question):

    answer = ask_question(question.question)

    return {
        "answer": answer
    }