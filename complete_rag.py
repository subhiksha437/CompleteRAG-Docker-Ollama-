"""
Complete Fixed RAG System V3
Fixes: PyPDF2, ChromaDB errors, cross-session memory
"""

import os
import json
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from persistent_memory import PersistentMemory

class ChatMemory:
    """Session-specific chat history"""
    def __init__(self, history_dir="./chat_history"):
        self.history_dir = history_dir
        os.makedirs(history_dir, exist_ok=True)
        self.session_file = None
        self.messages = []
        
    def new_session(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(self.history_dir, f"chat_{ts}.json")
        self.messages = []
        self._save()
        return ts
    
    def add(self, role: str, content: str):
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(msg)
        self._save()
    
    def _save(self):
        if self.session_file:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
    
    def get_context_str(self, n: int = 10) -> str:
        recent = self.messages[-n:] if self.messages else []
        if not recent:
            return ""
        
        ctx = []
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            ctx.append(f"{role}: {msg['content']}")
        return "\n".join(ctx)


class CompleteRAG:
    def __init__(self, backend="ollama", model_name=None):
        self.backend = backend
        self.model_name = model_name
        self.session_memory = ChatMemory()
        self.persistent_memory = PersistentMemory()  # NEW: Cross-session memory
        
        print("\n🚀 Initializing Complete RAG System V3...\n")
        
        # Embedding
        print("📥 Loading embedding model...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Vector DB - FIXED INITIALIZATION
        print("💾 Setting up ChromaDB...")
        self.chroma = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            self.collection = self.chroma.get_collection(name="docs")
            cnt = self.collection.count()
            print(f"✅ Loaded existing database with {cnt} documents")
        except:
            # Create with proper settings to avoid corruption
            self.collection = self.chroma.create_collection(
                name="docs",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Created new database")
        
        # LLM
        if backend == "ollama":
            self._init_ollama()
        else:
            self._init_huggingface()
        
        # Session
        sid = self.session_memory.new_session()
        print(f"💬 New session: {sid}")
        
        # Show persistent memory
        persistent_ctx = self.persistent_memory.get_context_str()
        if persistent_ctx:
            print("\n💾 Persistent Memory Loaded:")
            print(persistent_ctx)
        
        print("\n✨ System ready!\n")
    
    def _init_ollama(self):
        import requests
        try:
            r = requests.get('http://localhost:11434/api/tags', timeout=5)
            if r.status_code == 200:
                print(f"✅ Ollama connected: {self.model_name}")
                self.gen_func = self._ollama_gen
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            print(f"\n❌ Ollama not running!")
            print("Start it with:")
            print(f"  ollama serve")
            print(f"  ollama pull {self.model_name}")
            exit(1)
    
    def _init_huggingface(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"📥 Loading HuggingFace model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("✅ Model loaded")
        self.gen_func = self._hf_gen
    
    def _ollama_gen(self, prompt: str) -> str:
        import requests
        try:
            r = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "top_p": 0.9}
                },
                timeout=120
            )
            if r.status_code == 200:
                return r.json().get('response', '').strip()
            return f"Error: {r.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _hf_gen(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return resp[len(prompt):].strip()
    
    def retrieve_docs(self, query: str, top_k: int = 5) -> Tuple[List[str], List[str]]:
        """FIXED retrieval with error handling"""
        try:
            if self.collection.count() == 0:
                return [], []
            
            q_embed = self.embed_model.encode([query])
            
            results = self.collection.query(
                query_embeddings=q_embed.tolist(),
                n_results=min(top_k, self.collection.count())  # Don't exceed available docs
            )
            
            contexts = []
            sources = []
            
            if results and results.get('documents') and results['documents'][0]:
                contexts = results['documents'][0]
                if results.get('metadatas') and results['metadatas'][0]:
                    sources = [m.get('source', 'unknown') for m in results['metadatas'][0]]
            
            return contexts, sources
            
        except Exception as e:
            print(f"⚠️  Retrieval error (recovering): {e}")
            # Try to reset collection if corrupted
            try:
                self.chroma.delete_collection(name="docs")
                self.collection = self.chroma.create_collection(
                    name="docs",
                    metadata={"hnsw:space": "cosine"}
                )
                print("🔄 Database reset - please reload documents")
            except:
                pass
            return [], []
    
    def query(self, question: str) -> str:
        print(f"\n💭 Question: {question}\n")
        
        # Save user message
        self.session_memory.add("user", question)
        
        # Get contexts
        session_ctx = self.session_memory.get_context_str(n=10)
        persistent_ctx = self.persistent_memory.get_context_str()  # NEW: Always include
        
        # Retrieve documents
        doc_contexts = []
        doc_count = self.collection.count()
        
        if doc_count > 0:
            print(f"📖 Searching {doc_count} document chunks...")
            doc_contexts, sources = self.retrieve_docs(question, top_k=5)
            
            if doc_contexts:
                print(f"✅ Found {len(doc_contexts)} relevant chunks\n")
            else:
                print("ℹ️  No relevant documents\n")
        
        # Build prompt
        prompt = self._build_prompt(question, session_ctx, persistent_ctx, doc_contexts)
        
        # Generate
        print("🤖 Generating response...\n")
        response = self.gen_func(prompt)
        response = self._clean_response(response)
        
        # Save response
        self.session_memory.add("assistant", response)
        
        # Extract and store important info
        self.persistent_memory.extract_and_store(question, response)
        
        return response
    
    def _build_prompt(self, question: str, session_ctx: str, 
                      persistent_ctx: str, doc_contexts: List[str]) -> str:
        parts = []
        
        parts.append(
            "You are a helpful AI assistant that remembers information about the user "
            "across conversations."
        )
        
        # CRITICAL: Persistent memory first (name, preferences, etc.)
        if persistent_ctx:
            parts.append(f"\n{persistent_ctx}")
        
        # Current session context
        if session_ctx:
            parts.append(f"\n### Recent Conversation:\n{session_ctx}")
        
        # Documents
        if doc_contexts:
            doc_text = "\n\n---\n\n".join(doc_contexts[:5])
            parts.append(f"\n### Relevant Documents:\n{doc_text}")
        
        # Question
        parts.append(
            f"\n### User Question:\n{question}\n\n"
            "### Your Response:\n"
            "Answer naturally. If you know the user's name or preferences from persistent memory, "
            "use them. Be conversational and helpful."
        )
        
        return "\n".join(parts)
    
    def _clean_response(self, response: str) -> str:
        response = response.strip()
        for prefix in ["Answer:", "User:", "Assistant:", "AI:", "Human:", "Response:"]:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        return response
    
    def interactive(self):
        print("\n" + "="*70)
        print("🎯 COMPLETE RAG V3 - INTERACTIVE MODE")
        print("="*70)
        print("\nCommands:")
        print("  /memory    - Show persistent memory")
        print("  /forget    - Clear persistent memory")
        print("  /export    - Export current chat")
        print("  /stats     - Database statistics")
        print("  /clear     - New session (keeps persistent memory)")
        print("  /load      - Load documents")
        print("  /quit      - Exit")
        print("\n" + "="*70 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/quit":
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input == "/memory":
                    ctx = self.persistent_memory.get_context_str()
                    if ctx:
                        print(f"\n{ctx}\n")
                    else:
                        print("\n📝 No persistent memory stored yet\n")
                    continue
                
                elif user_input == "/forget":
                    confirm = input("Clear persistent memory? (yes/no): ")
                    if confirm.lower() == 'yes':
                        self.persistent_memory.clear()
                        print("✅ Persistent memory cleared\n")
                    continue
                
                elif user_input == "/export":
                    fname = self.session_memory.export(fmt='markdown')
                    print(f"\n✅ Exported to: {fname}\n")
                    continue
                
                elif user_input == "/stats":
                    cnt = self.collection.count()
                    msgs = len(self.session_memory.messages)
                    print(f"\n📊 Stats:")
                    print(f"   Documents: {cnt}")
                    print(f"   Session messages: {msgs}\n")
                    continue
                
                elif user_input == "/clear":
                    sid = self.session_memory.new_session()
                    print(f"\n✨ New session: {sid}")
                    print("💾 Persistent memory retained\n")
                    continue
                
                elif user_input == "/load":
                    self._load_documents()
                    continue
                
                # Normal query
                response = self.query(user_input)
                print(f"🤖 AI: {response}\n")
                print("-" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def _load_documents(self):
        print("\n📚 Load documents:")
        print("1. Single file")
        print("2. Folder")
        choice = input("Choose (1/2): ").strip()
        
        if choice == "1":
            path = input("File path: ").strip().strip('"')
            self._load_single_file(path)
        elif choice == "2":
            path = input("Folder path: ").strip().strip('"')
            self._load_folder(path)
    
    def _load_single_file(self, filepath: str):
        from add_docs import DocLoader
        loader = DocLoader(self.collection)
        chunks = loader.load_file(filepath)
        if chunks:
            filename = os.path.basename(filepath)
            loader.add_docs(chunks, [filename] * len(chunks))
            print(f"\n✅ Loaded {len(chunks)} chunks\n")
    
    def _load_folder(self, folder_path: str):
        from add_docs import DocLoader
        loader = DocLoader(self.collection)
        chunks, sources = loader.load_dir(folder_path)
        if chunks:
            loader.add_docs(chunks, sources)
            print(f"\n✅ Loaded {len(chunks)} chunks\n")
    
    def export(self, fmt='markdown', output_dir='./exports'):
        """Export session"""
        return self.session_memory.export(fmt, output_dir)


def main():
    print("\n" + "="*70)
    print("🚀 COMPLETE FIXED RAG V3")
    print("="*70 + "\n")
    
    # Backend selection
    print("Select LLM Backend:")
    print("1. Ollama (Recommended)")
    print("2. Hugging Face\n")
    
    choice = input("Choose (1/2): ").strip()
    
    if choice == "1":
        model = input("Model [llama3.2:3b]: ").strip() or "llama3.2:3b"
        rag = CompleteRAG(backend="ollama", model_name=model)
    else:
        model = input("Model [microsoft/phi-2]: ").strip() or "microsoft/phi-2"
        rag = CompleteRAG(backend="huggingface", model_name=model)
    
    rag.interactive()


if __name__ == "__main__":
    main()

