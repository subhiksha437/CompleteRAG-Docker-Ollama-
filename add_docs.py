"""Fixed Document Loader - PyPDF2 properly used"""
import os
from sentence_transformers import SentenceTransformer

class DocLoader:
    def __init__(self, collection=None):
        print("\n🚀 Initializing Document Loader...")
        self.embed = SentenceTransformer('all-MiniLM-L6-v2')
        
        if collection:
            self.coll = collection
        else:
            import chromadb
            self.chroma = chromadb.PersistentClient(path="./chroma_db")
            try:
                self.coll = self.chroma.get_collection(name="docs")
            except:
                self.coll = self.chroma.create_collection(
                    name="docs",
                    metadata={"hnsw:space": "cosine"}
                )
        
        print(f"✅ Connected ({self.coll.count()} documents)")
    
    def load_txt(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return self._chunk_text(content)
    
    def load_pdf(self, filepath: str):
        """FIXED PDF loading"""
        print(f"📕 Reading PDF...")
        try:
            import PyPDF2  # Explicit import
            
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                print(f"   Pages: {len(reader.pages)}")
                
                full_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                
                if not full_text.strip():
                    print("⚠️  No text extracted")
                    return []
                
                chunks = self._chunk_text(full_text)
                print(f"✅ Extracted {len(chunks)} chunks")
                return chunks
                
        except ImportError:
            print("❌ PyPDF2 not installed!")
            print("Run: pip install PyPDF2")
            return []
        except Exception as e:
            print(f"❌ PDF error: {e}")
            return []
    
    def load_docx(self, filepath: str):
        try:
            import docx
            doc = docx.Document(filepath)
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return self._chunk_text(full_text)
        except Exception as e:
            print(f"❌ DOCX error: {e}")
            return []
    
    def _chunk_text(self, text: str, size=500, overlap=100):
        chunks = []
        for i in range(0, len(text), size - overlap):
            chunk = text[i:i + size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def load_file(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"❌ Not found: {filepath}")
            return []
        
        print(f"\n📂 Loading: {os.path.basename(filepath)}")
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.txt':
            return self.load_txt(filepath)
        elif ext == '.pdf':
            return self.load_pdf(filepath)
        elif ext in ['.docx', '.doc']:
            return self.load_docx(filepath)
        else:
            print(f"⚠️  Unknown type: {ext}, trying as text...")
            return self.load_txt(filepath)
    
    def load_dir(self, dirpath: str):
        all_chunks = []
        all_sources = []
        
        print(f"\n📂 Scanning: {dirpath}")
        
        for root, _, files in os.walk(dirpath):
            for file in files:
                if file.endswith(('.txt', '.pdf', '.docx', '.doc')):
                    filepath = os.path.join(root, file)
                    chunks = self.load_file(filepath)
                    if chunks:
                        all_chunks.extend(chunks)
                        all_sources.extend([file] * len(chunks))
        
        print(f"\n✅ Total chunks: {len(all_chunks)}")
        return all_chunks, all_sources
    
    def add_docs(self, docs, sources=None):
        if not docs:
            return
        
        print(f"\n📚 Adding {len(docs)} chunks...")
        embeds = self.embed.encode(docs, show_progress_bar=True)
        
        existing = self.coll.count()
        ids = [f"doc_{existing + i}" for i in range(len(docs))]
        metas = [{"source": s} for s in sources] if sources else \
                [{"source": "manual"}] * len(docs)
        
        self.coll.add(
            embeddings=embeds.tolist(),
            documents=docs,
            metadatas=metas,
            ids=ids
        )
        
        print(f"✅ Added! Total: {self.coll.count()}")


def main():
    loader = DocLoader()
    
    while True:
        print("\n1. Add file")
        print("2. Add folder")
        print("3. Stats")
        print("4. Exit")
        
        choice = input("\nChoose: ").strip()
        
        if choice == "1":
            path = input("File path: ").strip().strip('"')
            chunks = loader.load_file(path)
            if chunks:
                loader.add_docs(chunks, [os.path.basename(path)] * len(chunks))
        
        elif choice == "2":
            path = input("Folder path: ").strip().strip('"')
            chunks, sources = loader.load_dir(path)
            if chunks:
                loader.add_docs(chunks, sources)
        
        elif choice == "3":
            print(f"\n📊 Documents: {loader.coll.count()}")
        
        elif choice == "4":
            break


if __name__ == "__main__":
    main()

