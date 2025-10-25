Here's comprehensive GitHub content for your RAG system:

---

# Enhanced RAG System V4 with Cross-Chat Memory 🧠

A sophisticated Retrieval-Augmented Generation (RAG) system that maintains persistent memory across chat sessions, enabling truly contextual conversations that remember user information and past interactions.

## 🌟 Key Features

### Cross-Chat Persistent Memory
- **Remembers Everything**: Information persists across all chat sessions, not just the current conversation
- **Semantic Search**: Find relevant facts from past conversations using advanced embeddings
- **User Profiling**: Automatically builds and maintains user profiles over time
- **Smart Context Building**: Combines current chat, past interactions, and document knowledge

### Dual-Memory Architecture
- **Session Memory**: Manages current conversation context
- **Persistent Memory**: Stores facts, user info, and interactions across all sessions
- **Automatic Extraction**: Intelligently identifies and stores important information

### Advanced RAG Capabilities
- **Vector Database**: ChromaDB for efficient document storage and retrieval
- **Semantic Embeddings**: SentenceTransformers for high-quality text representations
- **Multi-source Context**: Combines documents, chat history, and persistent facts
- **Flexible LLM Support**: Works with both Ollama and HuggingFace models

## 🛠️ Tech Stack

- **Python 3.8+**
- **LLM Backends**: Ollama / HuggingFace Transformers
- **Vector DB**: ChromaDB (persistent storage)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Dependencies**: PyTorch, requests, json

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama (optional, for local LLM) - [Installation Guide](https://ollama.ai)
- GPU recommended but not required

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/enhanced-rag-system.git
cd enhanced-rag-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch sentence-transformers chromadb transformers requests
```

4. **Install Ollama (Optional but Recommended)**
```bash
# Visit https://ollama.ai for installation
# Pull a model:
ollama pull llama3.2:3b
```

## 💻 Usage

### Basic Usage

```bash
python enhanced_rag.py
```

### Interactive Commands

Once running, you have access to these commands:

**Memory Management:**
- `/memory` - View all persistent memory from past chats
- `/add-fact` - Manually add important facts to memory
- `/add-info` - Add user profile information (name, job, etc.)
- `/search <query>` - Semantic search through past facts
- `/forget` - Clear all persistent memory

**Chat Management:**
- `/export` - Export current chat to markdown/JSON
- `/clear` - Start a new chat session
- `/stats` - View database statistics

**Document Management:**
- `/load` - Load documents (single file or folder)

**System:**
- `/quit` - Exit the application

### Example Interactions

```
You: My name is John and I work as a data scientist
🤖 AI: Nice to meet you, John! It's great to connect with a data scientist...

[Start new session]

You: What do you remember about me?
🤖 AI: You're John, and you work as a data scientist. I remember our previous conversation...
```

## 🏗️ Architecture

### Memory System

```
┌─────────────────────────────────────────┐
│         Enhanced RAG System              │
├─────────────────────────────────────────┤
│                                          │
│  ┌────────────────┐  ┌───────────────┐ │
│  │ Session Memory │  │  Persistent   │ │
│  │  (Current)     │  │  Memory       │ │
│  │                │  │  (All Chats)  │ │
│  └────────────────┘  └───────────────┘ │
│           │                  │          │
│           └─────────┬────────┘          │
│                     ▼                   │
│            ┌─────────────────┐          │
│            │  Context Builder │          │
│            └─────────────────┘          │
│                     │                   │
│                     ▼                   │
│            ┌─────────────────┐          │
│            │  LLM Generator  │          │
│            └─────────────────┘          │
└─────────────────────────────────────────┘
```

### Components

**PersistentMemory Class:**
- Stores facts with semantic embeddings
- Maintains user profile across sessions
- Provides semantic search capabilities
- Auto-saves to JSON files

**ChatMemory Class:**
- Manages current session history
- Exports conversations
- Loads context from past sessions

**CompleteRAG Class:**
- Orchestrates all components
- Handles document retrieval
- Manages LLM interactions
- Builds comprehensive context

## 📂 Project Structure

```
enhanced-rag-system/
├── enhanced_rag.py          # Main application
├── add_docs.py              # Document loader (optional)
├── persistent_memory/       # Cross-chat memory storage
│   ├── facts.json          # Stored facts with embeddings
│   ├── user_profile.json   # User information
│   └── interactions.json   # Interaction history
├── chat_history/           # Session-specific chats
│   └── chat_*.json        # Individual chat sessions
├── chroma_db/             # Vector database
└── exports/               # Exported conversations
```

## 🎯 Use Cases

1. **Personal AI Assistant**: Maintains context about your preferences, projects, and history
2. **Customer Support**: Remembers customer details across support sessions
3. **Educational Tutor**: Tracks learning progress and adapts to student needs
4. **Research Assistant**: Accumulates knowledge from documents and conversations
5. **Project Management**: Maintains context about ongoing projects and team members

## 🔧 Configuration

### LLM Backend Selection

**Ollama (Recommended):**
```python
rag = CompleteRAG(backend="ollama", model_name="llama3.2:3b")
```

**HuggingFace:**
```python
rag = CompleteRAG(backend="huggingface", model_name="microsoft/phi-2")
```

### Memory Settings

Adjust in `PersistentMemory.__init__()`:
```python
self.memory_dir = "./persistent_memory"  # Storage location
```

Adjust retrieval settings in queries:
```python
top_k = 5  # Number of relevant facts to retrieve
```

## 📊 Features Breakdown

### Semantic Search
- Uses cosine similarity for fact matching
- Embedding model: `all-MiniLM-L6-v2`
- Retrieves top-k most relevant facts

### Automatic Information Extraction
```python
keywords = ["remember", "my name", "I am", "I work", "I like", "I have"]
```
Automatically detects and stores user information based on keywords.

### Context Building
Combines multiple sources:
1. User profile from past chats
2. Related facts from semantic search
3. Recent conversation history
4. Relevant document chunks

## 🔒 Data Privacy

- All data stored locally
- No external API calls (except LLM)
- Full control over stored information
- Easy memory clearing with `/forget` command

## 🚧 Roadmap

- [ ] Add support for more embedding models
- [ ] Implement fact importance scoring
- [ ] Add conversation summarization
- [ ] Support for multimedia (images, audio)
- [ ] Web interface with Gradio/Streamlit
- [ ] Multi-user support
- [ ] Advanced NLP for better information extraction
- [ ] Integration with external knowledge bases

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Requirements

```txt
torch>=2.0.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
transformers>=4.30.0
requests>=2.28.0
```

## 🐛 Troubleshooting

**Ollama Connection Error:**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is installed
ollama list
```

**Memory Issues:**
- Clear persistent memory: `/forget` command
- Reduce `top_k` parameter for retrieval
- Use smaller embedding model

**ChromaDB Errors:**
- Delete `chroma_db/` folder and restart
- Check disk space

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Developed as a demonstration of advanced RAG techniques with persistent memory capabilities.

## 🙏 Acknowledgments

- SentenceTransformers team for embedding models
- ChromaDB team for vector database
- Ollama team for local LLM infrastructure
- HuggingFace for transformer models

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ **If you find this project useful, please consider giving it a star!**

---

## 🎓 Learn More

- [RAG Explained](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)

---

**Made with ❤️ for the AI community**
