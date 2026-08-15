
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# The FAISS index is needed at runtime. Choose ONE of:
#   A) Commit your local `faiss_index/` to the repo (simplest), OR
#   B) Build it inside the image (requires MEDICAL_PDF_PATH + internet for the
#      embedding model). Uncomment the next line and set MEDICAL_PDF_PATH:
# RUN python -m app.loaders

EXPOSE 8000
# Render (and most hosts) inject the listening port via $PORT.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
