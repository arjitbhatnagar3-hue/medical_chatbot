"""
Builds the FAISS vector index from the medical PDF.

Located in the `app/` package. Run ONCE (from the project root) after setting
MEDICAL_PDF_PATH in .env:
    python -m app.loaders
    # or:  python app/loaders.py

The resulting `faiss_index/` folder (at the PROJECT ROOT) is what app/rag.py
loads at runtime.
"""
import os
from pathlib import Path
import time 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
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
HF_TOKEN = os.getenv("HF_TOKEN")


class BatchEndpointEmbeddings(HuggingFaceEndpointEmbeddings):
    """Embeds in small batches with retry, so we never send all ~22k chunks in
    one HF Inference API request (which times out with a 504)."""

    def embed_documents(self, texts, batch_size=256, max_retries=4):
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for attempt in range(max_retries):
                try:
                    vectors.extend(super().embed_documents(batch))
                    break
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s backoff
            time.sleep(0.2)  # small gap between batches to avoid rate limits
        return vectors


def build_index(pdf_path: str = PDF_PATH, index_dir: Path = INDEX_DIR) -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # API-based embeddings, batched (same method rag.py uses at query time).
    embeddings = BatchEndpointEmbeddings(
        repo_id=EMBEDDING_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
        task="feature-extraction",
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Saved FAISS index to {index_dir}")


if __name__ == "__main__":
    build_index()
