"""
Builds the FAISS vector index from the medical PDF (LOCAL embeddings).

Located in the app/ package. Run ONCE (from the project root):
    python -m app.loaders

The resulting faiss_index/ (at the PROJECT ROOT) is what app/rag.py loads at runtime.
Only needed if you don't already have a faiss_index/ folder.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# This file is inside app/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = os.getenv(
    "MEDICAL_PDF_PATH",
    r"C:\Users\sharm\AppData\Roaming\Microsoft\Windows\Network Shortcuts\medical_book_for_project.pdf",
)
INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", str(PROJECT_ROOT / "faiss_index")))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(pdf_path: str = PDF_PATH, index_dir: Path = INDEX_DIR) -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Saved FAISS index to {index_dir}")


if __name__ == "__main__":
    build_index()
