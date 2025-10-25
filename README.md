# Complete RAG V3 - Docker Edition

## Quick Start

### 1. Build
```powershell
.\build.ps1
```

### 2. Start (with Ollama)
```powershell
.\start.ps1
```

### 3. Start (HuggingFace only)
```powershell
.\start-hf.ps1
```

### 4. Load Documents
```powershell
.\load-docs.ps1
```

### 5. Stop
```powershell
.\stop.ps1
```

## Manual Commands

### Build images
```bash
docker-compose build
```

### Start Ollama server
```bash
docker-compose up -d ollama
```

### Pull Ollama model
```bash
docker exec ollama-server ollama pull llama3.2:3b
```

### Run RAG system
```bash
docker-compose run --rm rag-system
```

### Load documents
```bash
docker-compose run --rm rag-system python add_docs.py
```

## Data Persistence

All data is stored in local folders:
- ./documents/ - Your documents
- ./chroma_db/ - Vector database
- ./chat_history/ - Chat sessions
- ./exports/ - Exported chats
- ./persistent_memory/ - User memory

## Ollama Connection

The RAG system connects to Ollama at http://ollama:11434 inside Docker network.

## Troubleshooting

### Ollama not connecting
```bash
docker exec -it ollama-server ollama list
```

### View logs
```bash
docker-compose logs -f
```

### Reset everything
```powershell
.\cleanup.ps1
```
