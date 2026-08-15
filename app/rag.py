"""
RAG engine for the medical chatbot.

Responsibilities:
  - Load the FAISS index built by loaders.py
  - Retrieve the most relevant chunks for a question
  - Build a (system + human) prompt and call the LLM via the HF Inference API
  - Return the answer together with its source citations

Initialization (model + index load) happens once at import time, but any
failure is captured in `_init_error` instead of crashing the import, so the
API can still start and report a clear health status.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint

load_dotenv()

# --- Paths (resolved relative to the project, not the current working dir) ---
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", str(BASE_DIR.parent / "faiss_index")))

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

# FAISS IndexFlatL2 distance: 0 = identical, up to ~4 for opposite directions.
# If the BEST match is farther than this, we treat the question as unanswerable
# from the book. Tune this number for your data; set it very high to disable.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "1.5"))

SYSTEM_PROMPT = (
    "You are a careful medical assistant.\n"
    "Answer the question ONLY using the provided context.\n"
    "Do not use outside knowledge.\n"
    "If the answer cannot be found in the context, say exactly: "
    '"I don\'t know based on the provided information."\n'
    'End every answer with this disclaimer: '
    '"This information is not a substitute for professional medical advice."'
)

HUMAN_TEMPLATE = """Context:
{context}

Question:
{question}

Answer:"""


_model = None
_db = None
_init_error = None


def _initialize() -> None:
    """Load the LLM and FAISS index once. Captures errors instead of crashing."""
    global _model, _db, _init_error
    try:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN is not set. Add it to your .env file.")

        if not INDEX_DIR.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_DIR}. Run loaders.py to build it first."
            )

        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            huggingfacehub_api_token=HF_TOKEN,
            task="text-generation",
            temperature=0.2,
            max_new_tokens=512,
        )
        _model = ChatHuggingFace(llm=llm)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _db = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:  # captured on purpose
        _init_error = exc


_initialize()


def ask_question(question: str) -> dict:
    """Return {'answer': str, 'sources': list[dict]} for a user question."""
    if _init_error is not None:
        raise RuntimeError(str(_init_error))

    results = _db.similarity_search_with_score(question, k=2)
    docs = [doc for doc, _score in results]

    # Gate: if the closest chunk is still too far away, say we don't know.
    best_score = results[0][1] if results else float("inf")
    if best_score > SCORE_THRESHOLD:
        return {"answer": "I don't know based on the provided information.", "sources": []}

    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=HUMAN_TEMPLATE.format(context=context, question=question)),
    ]
    response = _model.invoke(messages)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
        }
        for doc in docs
    ]
    return {"answer": response.content, "sources": sources}
