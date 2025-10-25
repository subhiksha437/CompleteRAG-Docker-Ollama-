# Start RAG system in Docker
Write-Host "
🚀 Starting RAG system..." -ForegroundColor Cyan

# Start Ollama first (if using)
Write-Host "Starting Ollama server..." -ForegroundColor Yellow
docker-compose up -d ollama

Start-Sleep -Seconds 3

# Pull Ollama model (optional)
Write-Host "
Pulling Ollama model..." -ForegroundColor Yellow
docker exec ollama-server ollama pull llama3.2:3b

# Start RAG system
Write-Host "
🤖 Starting RAG system..." -ForegroundColor Cyan
docker-compose run --rm rag-system
