# Dockerfile for RAG System
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Python files
COPY persistent_memory.py .
COPY complete_rag.py .
COPY add_docs.py .

# Install Python packages
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers \
    accelerate \
    chromadb \
    sentence-transformers \
    PyPDF2 \
    python-docx \
    requests

# Create necessary directories
RUN mkdir -p /app/documents \
    /app/chroma_db \
    /app/chat_history \
    /app/exports \
    /app/persistent_memory

EXPOSE 8000

CMD ["python", "complete_rag.py"]
