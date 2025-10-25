# Load documents into RAG system
Write-Host "
📚 Starting document loader..." -ForegroundColor Cyan
docker-compose run --rm rag-system python add_docs.py
